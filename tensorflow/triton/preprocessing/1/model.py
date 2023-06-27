import triton_python_backend_utils as pb_utils
import numpy as np
# from skimage.transform import resize

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            raw_images = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()

            # raw_images = resize(raw_images, (512,512))
            img_tensor = np.expand_dims(raw_images,axis=0)
            img_tensor /= 255.
            input_image_tensor = pb_utils.Tensor(
                "input_image", img_tensor.astype(np.float32)
            )
            response = pb_utils.InferenceResponse(
                output_tensors = [input_image_tensor]
            )

            responses.append(response)

        return responses
            