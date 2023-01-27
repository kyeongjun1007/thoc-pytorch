from THOCBase import THOCBase

class THOCValidator(THOCBase) :
    def __init__(self, model, model_params, logger_params, run_params):
        super(THOCValidator, self).__init__(model_params, logger_params, run_params)
        self.model = model

