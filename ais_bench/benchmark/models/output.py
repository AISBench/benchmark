import os
import time
from abc import abstractmethod
from typing import Union

from PIL import Image

import numpy as np


class Output:
    def __init__(self, perf_mode: bool = False) -> None:
        self.perf_mode = perf_mode
        self.success: bool = False
        self.error_info: str = ""
        self.time_points: list[float] = []
        self.content: list[str] | str = ""
        self.reasoning_content: list[str] | str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.extra_perf_data: dict = {}
        self.extra_details_data: dict = {}
        self.input: list | str = None
        self.uuid: str = ""
        # A unique identifier for each case:
        # In multi-turn dialogue scenarios, all turns of the same sample share the same uuid.
        # In pass@k scenarios, the same sample is sampled k times and each run receives a distinct uuid
        self.turn_id: int = 0

    @abstractmethod
    def get_metrics(self) -> dict:
        """Calculate and return performance metrics for the output.

        Returns:
            dict: Cleaned metrics dictionary with performance data
        """
        pass

    def update_extra_perf_data_from_stream_response(
        self, stream_response: dict
    ) -> None:
        """Update the extra perf data from response.

        Args:
            stream_response: Stream response
        """
        pass

    def update_extra_perf_data_from_text_response(self, text_response: dict) -> None:
        """Update the extra perf data from text response.

        Args:
            text_response: Text response
        """
        pass

    def update_extra_details_data_from_stream_response(
        self, stream_response: dict
    ) -> None:
        """Update the extra details data from stream response.

        Args:
            stream_response: Stream response
        """
        pass

    def update_extra_details_data_from_text_response(self, text_response: dict) -> None:
        """Update the extra details data from text response.

        Args:
            text_response: Text response
        """
        pass

    def _concate_reasoning_content(self, content, reasoning_content) -> str:
        """Concatenate reasoning content with main content.

        Args:
            content: Main content string
            reasoning_content: Reasoning content string

        Returns:
            str: Combined content with reasoning
        """
        if reasoning_content:
            if content:
                return reasoning_content + "</think>" + content
            else:
                return reasoning_content
        else:
            return content

    def get_prediction(self) -> dict:
        """Get the final prediction by combining content and reasoning.

        Returns:
            dict: Combined prediction content
        """
        if not self.reasoning_content:
            return self.content

        if isinstance(self.content, list) and isinstance(self.reasoning_content, list):
            return [
                self._concate_reasoning_content(content, reasoning_content)
                for content, reasoning_content in zip(
                    self.content, self.reasoning_content
                )
            ]
        elif isinstance(self.reasoning_content, str):
            return self._concate_reasoning_content(self.content, self.reasoning_content)

        return self.content

    def to_dict(self):
        """Convert all instance attributes to dictionary.

        Returns:
            dict: Dictionary containing all instance attributes
        """
        return self.__dict__

    async def record_time_point(self) -> None:
        """Record a time point for performance measurement.

        This method is called by the model to record timing data.
        """
        if self.perf_mode:
            self.time_points.append(time.perf_counter())

    async def clear_time_points(self) -> None:
        """Clear the time points for performance measurement.

        This method is called by the model to clear the time points.
        """
        self.time_points = []


class RequestOutput(Output):

    def get_metrics(self) -> dict:
        """Calculate and return detailed performance metrics for request output.

        Returns:
            dict: Enhanced metrics dictionary with request-specific performance data
        """

        def clean_result(res):
            for key in ["content", "reasoning_content", "perf_mode"]:
                res.pop(key, None)
            return res

        self.prediction = self.get_prediction()
        self.time_points = np.array(self.time_points, dtype=np.float64)
        if not self.success:
            result = clean_result(self.to_dict())
            return result

        if self.time_points.size <= 1:
            self.success = False
            self.error_info = "chunk size is less than 2"
        result = clean_result(self.to_dict())
        return result


class FunctionCallOutput(Output):

    def __init__(self, perf_mode: bool = False) -> None:
        super().__init__(perf_mode)
        self.inference_log: list[dict] = []
        self.tool_calls: list[dict] = []

    def update_extra_details_data_from_text_response(self, text_response: dict) -> None:
        """Update the extra details data from text response.

        Args:
            text_response: Text response
        """
        for item in text_response.get("choices", []):
            message = item.get("message", {})
            self.extra_details_data["message"] = message
            return  # only one message is allowed


LLM_META_DATA_TYPE = Union[Image, str]


class LMMOutput(Output):
    def __init__(self, perf_mode: bool = False) -> None:
        super().__init__(perf_mode)
        self.content: list[LLM_META_DATA_TYPE] = [""]
        self.HANDLER_MAP = {
            Image.Image: self._handle_image,
            str: self._handle_text,
        }

    def get_prediction(self, save_dir: str) -> dict:
        """Get the final prediction by combining content and reasoning.

        Returns:
            dict: Combined prediction content
        """
        output = []
        for i, item in enumerate(self.content):
            output.append(self.HANDLER_MAP[type(item)](save_dir, i))
        if len(output) == 1:
            return output[0]
        else:
            return output

    def _handle_image(self, save_dir: str, index: int) -> str:
        """Handle image content.

        Args:
            save_dir: Directory to save image
            index: Index of image in content list

        Returns:
            str: Last two levels of image path
        """
        image = self.content[index]
        image_path = os.path.join(save_dir, f"image_{self.uuid}_{index}.png")
        if os.path.exists(image_path):
            os.remove(image_path)
        image.save(image_path)
        return os.path.join(*image_path.split(os.sep)[-2:])

    def _handle_text(self, save_dir: str, index: int) -> str:
        """Handle text content.

        Args:
            save_dir: Directory to save text
            index: Index of text in content list

        Returns:
            str: Text content
        """
        return self.content[index]


