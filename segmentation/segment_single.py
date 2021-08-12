from base import BaseLinearFlow
import metaflow.package
from metaflow import step, batch, retry, Parameter, S3, conda

class SegmentationSingleStepFlow(BaseLinearFlow):

    model_path = Parameter('model_path',
                            default='s3://kespry-metaflow/model/best_model_unet_fullmask_epoch26.pth',
                            help='Path To the model on S3.')
    
    mission_id = Parameter('mission_id',
                            default='173567',
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
            from PIL import Image
            from skimage.io import imsave
            from kespryml_roof_wireframe.inference import WireframeNet

            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        except Exception as e:
            print('Not Able to import: ' + e)

        try:
            with S3() as s3:
                model_path = s3.get(self.model_path).path
                self.model = WireframeNet(model_weight_path=model_path)
        except Exception as e:
            print('Not able to load the model.' + e)
        
        print("Model Loaded...")

        try:
            with S3() as s3:
                ortho_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'orthomosaic.tif')).path
                dsm_color_image_path = s3.get(os.path.join(self.source_dir, self.mission_id, 'preview_products', 'dsm_colored.tif')).path

                os.system('gdal_translate -of GTiff -r cubicspline -outsize {} {} {} {}'.format(800, 1024, ortho_image_path, os.path.join(os.path.dirname('__file__'), 'orthomosaic_resized.tif')))
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'orthomosaic_resized.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'orthomosaic_resized.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'orthomosaic_resized.tif')))

                os.system('gdal_translate -of GTiff -r cubicspline -outsize {} {} {} {}'.format(800, 1024, dsm_color_image_path, os.path.join(os.path.dirname('__file__'), 'dsm_colored_resized.tif')))
                file_stream = io.BytesIO()
                im = Image.open(os.path.join(os.path.dirname('__file__'), 'dsm_colored_resized.tif'))
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'dsm_colored_resized.tif'), file_stream.getvalue(), overwrite=True)
                os.system('rm -rf {}'.format(os.path.join(os.path.dirname('__file__'), 'dsm_colored_resized.tif')))
                ortho_image_path = s3.get(os.path.join(self.save_path, 'orthomosaic_resized.tif')).path
                dsm_color_image_path = s3.get(os.path.join(self.save_path, 'dsm_colored_resized.tif')).path

                self.mask = self.model.predict(ortho_image_path=ortho_image_path, dsm_col_image_path=dsm_color_image_path)

                file_stream = io.BytesIO()
                im = Image.fromarray(self.mask)
                im.save(file_stream, format='tiff')
                s3.put(os.path.join(self.save_path, 'output.tif'), file_stream.getvalue(), overwrite=True)
                

        except Exception as e:
            print('Not able preprocess Image.'+ e)

        print('Finished Processing.')
        self.next(self.end)

    @step
    def end(self):
        print('End')


if __name__ == '__main__':
    SegmentationSingleStepFlow()
