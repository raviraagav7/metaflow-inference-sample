import metaflow.package
from metaflow import step, batch, retry, Parameter, S3, FlowSpec

class WireframeSegmentationFlow(FlowSpec):

    boundary_mask_model_path = Parameter('boundary_mask_model_path',
                            default='s3://kespry-metaflow/model/best_model_unet_fullmask_epoch26.pth',
                            help='Path To the boundary mask model on S3.')

    roof_edge_mask_model_path = Parameter('roof_edge_mask_model_path',
                            default='s3://kespry-metaflow/model/best_model_unet_v1.pth',
                            help='Path To the edge mask model on S3.')

    mission_id = Parameter('mission_id',
                            default='110636',
                            help='Mission Id')

    source_dir = Parameter('source_dir',
                            default='s3://kespry-files/images',
                            help='Source Image')

    save_path = Parameter('save_path',
                            default='s3://kespry-metaflow/processed_image',
                            help='Save Path.') 

    
    @batch(image='metaflow-example:latest')
    @step
    def start(self):
        """
        The 'start' step is a regular step, so runs locally on the machine from
        which the flow is executed.

        """
        try:
            import os
            import io
            import gdal
            import json
            from PIL import Image
            from kespryml_roof_wireframe.inference import WireframeNet
            from kespryml_roof_wireframe.utils import generate_richdem
            from kespryml_roof_wireframe.image2geojson import Image2GeoJson, ProcessThinSpec
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        except Exception as e:
            print('Not Able to import: {}'.format(e))

        try:
            with S3() as s3:

                roof_edge_model_path = s3.get(self.roof_edge_mask_model_path).path
                self.roof_edge_model = WireframeNet(model_weight_path=roof_edge_model_path, new_model_flag=True)

                boundary_model_path = s3.get(self.boundary_mask_model_path).path
                self.boundary_model = WireframeNet(model_weight_path=boundary_model_path, new_model_flag=False)

        except Exception as e:
            print('Not able to load the model: {}'.format(e))
            raise ValueError
        
        print("Model Loaded...")

        # Preprocess Step
        try:
            with S3() as s3:

                ortho_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'orthomosaic.tif')).path
                dsm_color_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'dsm_colored.tif')).path

                # Resize Ortho Image to fixed size 800, 1024 and save on S3.
                os.system('gdal_translate -of GTiff -r cubicspline -outsize {} {} {} {}'.format(800, 1024, ortho_image_path, os.path.join(os.path.dirname('__file__'), 'orthomosaic_resized.tif')))
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'orthomosaic_resized.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'orthomosaic_resized.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'orthomosaic_resized.tif')))
                
                # Resize DSM Color Image to fixed size 800, 1024 and save on S3.
                os.system('gdal_translate -of GTiff -r cubicspline -outsize {} {} {} {}'.format(800, 1024, dsm_color_image_path, os.path.join(os.path.dirname('__file__'), 'dsm_colored_resized.tif')))
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'dsm_colored_resized.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'dsm_colored_resized.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'dsm_colored_resized.tif')))

                dsm_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'dsm.tif')).path
                gdal_src = gdal.Open(dsm_image_path)
                gdal_array = gdal_src.ReadAsArray()
                height, width = gdal_array.shape

                # Convert DSM Image to Byte DSM Image and save on S3.
                os.system('gdal_translate -co COMPRESS=LZW -ot Byte -scale {} {}'.format(dsm_image_path, os.path.join(os.path.dirname('__file__'), 'dsm_byte.tif')) )
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'dsm_byte.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'dsm_byte.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'dsm_byte.tif')))

                # Resize Ortho Image to original DSM Image size and save on S3.
                os.system('gdal_translate -of GTiff -r cubicspline -outsize {} {} {} {}'.format(width, height, ortho_image_path, os.path.join(os.path.dirname('__file__'), 'orthomosaic_original_resized.tif')))
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'orthomosaic_original_resized.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'orthomosaic_original_resized.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'orthomosaic_original_resized.tif')))

                # Generate RichDEM Image and save on S3.
                generate_richdem(dsm_image_path, os.path.join(os.path.dirname('__file__'), 'richdem.tif'))
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'richdem.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'richdem.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'richdem.tif')))


        except Exception as e:
            print('Not able preprocess Image: {}'.format(e))

        # Predict Step
        try:
            with S3() as s3:

                # Predict the roof edge mask and save on S3.
                ortho_orig_image_path = s3.get(os.path.join(self.save_path, 'orthomosaic_original_resized.tif')).path
                dsm_byte_image_path = s3.get(os.path.join(self.save_path, 'dsm_byte.tif')).path
                richdem_image_path = s3.get(os.path.join(self.save_path, 'richdem.tif')).path

                roof_edge_mask = self.roof_edge_model.predict(richdem_image_path=richdem_image_path, dsm_byte_image_path=dsm_byte_image_path, ortho_orig_image_path=ortho_orig_image_path)

                file_stream = io.BytesIO()
                im = Image.fromarray(roof_edge_mask)
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'output_roof_edge.tif'), file_stream.getvalue(), overwrite=True)

                # Predict the boundary mask and save on S3.
                ortho_image_path = s3.get(os.path.join(self.save_path, 'orthomosaic_resized.tif')).path
                color_dsm_image_path = s3.get(os.path.join(self.save_path, 'dsm_colored_resized.tif')).path

                boundary_mask = self.boundary_model.predict(ortho_image_path=ortho_image_path, color_dsm_image_path=color_dsm_image_path)

                file_stream = io.BytesIO()
                im = Image.fromarray(boundary_mask)
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'output_boundary.tif'), file_stream.getvalue(), overwrite=True)
        except Exception as e:
            print('Not able Predict Image: {}'.format(e))

        # Post Process Step

        try:
            with S3() as s3:
                
                dsm_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'dsm.tif')).path
                ortho_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'orthomosaic.tif')).path


                Image2GeoJson().run_from_path(dsm_path=dsm_image_path,
                                      ortho_path=ortho_image_path,
                                      roof_path=s3.get(os.path.join(self.save_path, 'output_boundary.tif')).path,
                                      wireframe_path=s3.get(os.path.join(self.save_path, 'output_roof_edge.tif')).path,
                                      specification=ProcessThinSpec(),
                                      save_geojson_path=os.path.join(os.path.dirname('__file__'), 'output.geojson'),
                                      save_overlay_img_path=os.path.join(os.path.dirname('__file__'), 'output_overlay.tif'))
                
                with open(os.path.join(os.path.dirname('__file__'), 'output.geojson')) as f:
                    json_data = json.dumps(json.load(f))
                s3.put(os.path.join(self.save_path, 'output.geojson'), json_data)

                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'output_overlay.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'output_overlay.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'output_overlay.tif')))

        except Exception as e:
            print('Not able Post Process Image: {}'.format(e))

        print('Finished Processing.......')

        self.next(self.end)

    @step
    def end(self):
        """
        The 'end' step is a regular step, so runs locally on the machine from
        which the flow is executed.

        """
        print("AWS Task is finished.")


if __name__ == '__main__':
    WireframeSegmentationFlow()
