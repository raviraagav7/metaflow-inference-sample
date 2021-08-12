from metaflow import FlowSpec, step

class BaseLinearFlow(FlowSpec):
    
    @step
    def start(self):
        """
        The 'start' step is a base step.

        """
        print('This is base method for start step.')
        self.next(self.process)
       
    @step
    def process(self):
        """
        The 'process' step is a base step.

        """
        print('This is base method for process step.')
        self.next(self.end)
    
    @step
    def end(self):
        """
        The 'end' step is a base step.

        """
        print('This is base method for end step.')

if __name__ == '__main__':
    BaseLinearFlow()

        