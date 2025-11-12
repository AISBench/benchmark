BFCL_INSTALLED = True
try:
    from bfcl_eval.constants.eval_config import PROMPT_PATH
    from bfcl_eval.utils import (
        is_empty_output,
        is_function_calling_format_output,
        is_java,
        is_js,
    )

    from bfcl_eval.utils import make_json_serializable
    from bfcl_eval._llm_response_generation import process_multi_turn_test_case

    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
        is_empty_execute_response,
    )
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.model_style import ModelStyle
    from bfcl_eval.model_handler.utils import (
        func_doc_language_specific_pre_processing,
        convert_to_tool,
        convert_to_function_call,
        default_decode_ast_prompting,
        default_decode_execute_prompting,
        system_prompt_pre_processing_chat_model,
        format_execution_results_prompting,
    )
    from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
    from bfcl_eval.constants.default_prompts import (
        MAXIMUM_STEP_LIMIT,
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
        DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING,
    )
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
        execute_multi_turn_func_call,
    )
except ImportError as e:
    BFCL_INSTALLED = False