import numpy as np
import sys
import os
import json
from pathlib import Path

import torch

import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    # Every Python model must have "TritonPythonModel" as the class name!
    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args['model_config']), "output"
            )['data_type']
        )

        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda()
        print("TritonPythonModel initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            input_ids = input_ids.as_numpy()
            input_ids = torch.as_tensor(input_ids).long().cuda()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            attention_mask = attention_mask.as_numpy()
            attention_mask = torch.as_tensor(attention_mask).long().cuda()
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            summary = self.model.generate(**inputs, max_length=256, num_beams=1)
            # Convert to numpy array on cpu:
            np_summary = summary.cpu().int().detach().numpy()
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "output",
                        np_summary.astype(self.output_dtype)
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print("TritonPythonModel finalized")
