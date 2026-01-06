import os
import json
import time
import google.generativeai as genai
from github import Github
from github import Auth
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure the genai client with the API key
# Handle both GOOGLE_API_KEY and GEMINI_API_KEY for compatibility
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Set the API key that genai package expects
api_key = GOOGLE_API_KEY or GEMINI_API_KEY
if not api_key:
    raise ValueError("Either GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set")

genai.configure(api_key=api_key)

# Initialize GitHub client
auth = Auth.Token(os.getenv('GITHUB_TOKEN'))
github = Github(auth=auth)

# Load GitHub event data
event_path = os.getenv('GITHUB_EVENT_PATH')
with open(event_path, 'r') as f:
    event = json.load(f)

# Determine event type and get issue information
event_name = os.getenv('GITHUB_EVENT_NAME')

if event_name == 'issues':
    # For issues event
    issue_number = event['issue']['number']
    issue_title = event['issue']['title']
    issue_body = event['issue']['body']
    issue_labels = [label['name'] for label in event['issue']['labels']]
    repo_full_name = event['repository']['full_name']
elif event_name == 'issue_comment':
    # For issue_comment event
    issue_number = event['issue']['number']
    issue_title = event['issue']['title']
    issue_body = event['issue']['body']
    issue_labels = [label['name'] for label in event['issue']['labels']]
    repo_full_name = event['repository']['full_name']
    comment_body = event['comment']['body']
else:
    # Unsupported event type
    print(f"Unsupported event type: {event_name}")
    exit(0)

# Determine the issue type based on title or labels
issue_type = "bug"  # Default to bug
if any(label in ['enhancement', 'feature', 'Feature'] for label in issue_labels):
    issue_type = "feature"
elif any(label in ['doc', 'documentation', 'Documentation'] for label in issue_labels):
    issue_type = "documentation"
elif any(label in ['question', 'consult', 'Consult'] for label in issue_labels):
    issue_type = "consult"

# Define required sections based on issue type
required_sections = {
    "bug": {
        "zh": [
            "操作系统及版本",
            "安装工具的python环境",
            "python版本",
            "AISBench工具版本",
            "AISBench执行命令",
            "模型配置文件或自定义配置文件内容",
            "实际行为"
        ],
        "en": [
            "Operating System and Version",
            "Python Environment for Tool Installation",
            "Python Version",
            "AISBench Tool Version",
            "AISBench Execution Command",
            "Model Configuration File or Custom Configuration File Content",
            "Actual Behavior"
        ]
    },
    "feature": {
        "zh": ["问题/痛点描述", "建议方案", "预期价值"],
        "en": ["Problem/Pain Point Description", "Proposed Solution", "Expected Value"]
    },
    "documentation": {
        "zh": ["文档位置（可指定多个文档链接）", "当前内容描述", "修改建议"],
        "en": ["Documentation Location (Multiple document links can be specified)", "Current Content Description", "Modification Suggestion"]
    },
    "consult": {
        "zh": ["疑问描述"],
        "en": ["Inquiry Description"]
    }
}

# Detect language of the issue body
def detect_language(text):
    if re.search(r'[\u4e00-\u9fa5]', text):
        return "zh"
    return "en"

language = detect_language(issue_body)

# Get the appropriate sections for the detected language and issue type
sections_to_check = required_sections.get(issue_type, {}).get(language, required_sections["bug"][language])

# Create prompt for Gemini API
prompt = f"""
You are an assistant that checks if GitHub issue content is complete based on the required sections.

Issue Title: {issue_title}
Issue Body:
{issue_body}

Required Sections ({language}):
{chr(10).join([f"- {section}" for section in sections_to_check])}

Please check if the issue contains all the required sections with sufficient information. For each section:
1. Indicate if it's present and complete
2. If not complete, specify what information is missing or needs to be补充 (in {language})
3. Ensure the content is relevant to the section title and not meaningless text (e.g., placeholder text, repeated characters, unrelated content)

Format your response as follows:

## 问题内容检查结果

### 检查状态
[PASS/FAIL]

### 详细检查
{chr(10).join([f"- {section}: [COMPLETE/INCOMPLETE]" for section in sections_to_check])}

### 改进建议
[List specific suggestions for each incomplete section, or "所有内容已完备！"]

### 补充说明
[Any additional comments]

Please respond in {language} and ensure your response is clear and helpful.
"""

# Call Gemini API
try:
    # Create a model instance
    model = genai.GenerativeModel('gemini-2.5-flash')

    # Generate content using the model
    # Implement retry mechanism
    max_retries = 3
    retry_delay = 1  # Initial delay in seconds
    retry_count = 0
    response = None

    # Network-related error patterns to retry on
    network_error_patterns = [
        "network", "timeout", "connection", "connect", "refused", "reset",
        "closed", "unreachable", "error 429", "error 500", "error 502",
        "error 503", "error 504", "server error"
    ]

    # Initialize check_status variable
    check_status = None

    while retry_count < max_retries:
        try:
            response = model.generate_content(
                contents=prompt,
                generation_config={
                    'temperature': 0.3,
                    'top_p': 1.0,
                    'top_k': 1,
                    'max_output_tokens': 1024
                }
            )
            break  # Success, exit the loop
        except Exception as e:
            error_msg = str(e).lower()

            # Check if the error is related to input token limits
            token_limit_patterns = [
                "token limit exceeded", "input too long", "context window exceeded",
                "max input tokens", "exceeds the maximum", "content too long"
            ]
            is_token_limit_error = any(pattern in error_msg for pattern in token_limit_patterns)

            if is_token_limit_error:
                # Handle input token limit error specially
                if language == "zh":
                    comment_body = "🤖 基于AI机器人的issue内容完整性检查结果:\n\n⚠️ 由于issue内容过长，超出了AI模型的处理能力，无法进行详细检查。默认视为内容完整。\n👉 如果想重新检查，请尝试简化issue内容后在评论区@issue_checker即可。"
                else:
                    comment_body = "🤖 issue content check result from AI robot:\n\n⚠️ The issue content is too long, exceeding the AI model's processing capacity. Cannot perform detailed check. Defaulting to content complete.\n👉 If you want to re-check, please try simplifying the issue content and comment @issue_checker."
                check_status = "PASS"  # Set status to PASS for token limit error
                break  # Exit the loop with the token limit message

            # Check if the error is network-related
            is_network_error = any(pattern in error_msg for pattern in network_error_patterns)

            if not is_network_error:
                raise  # Re-raise non-network errors immediately

            retry_count += 1
            if retry_count >= max_retries:
                raise  # Re-raise if max retries exceeded

            # Exponential backoff
            delay = retry_delay * (2 ** (retry_count - 1))
            print(f"Network error occurred. Retrying in {delay} seconds... (Attempt {retry_count}/{max_retries})")
            time.sleep(delay)

    if response is not None and hasattr(response, 'text'):
        if language == "zh":
            comment_body = "🤖 基于AI机器人的issue内容完整性检查结果:\n\n" + response.text + "\n\n👉 如果想重新检查，在评论区@issue_checker即可。\n\n" \
                + "【强烈推荐❤️‍🔥】确保issue描述完整后，可以试着将issue交给[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AISBench/benchmark)回答，deepwiki包含了和工具相关的所有知识库"
        else:
            comment_body = "🤖 issue content check result from AI robot:\n\n" + response.text + "\n\n👉 If you want to re-check, please comment @issue_checker. \n\n" \
                + "[Strongly recommended❤️‍🔥]Ensure your issue description is complete, then try to ask [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/AISBench/benchmark) for help, as deepwiki contains all the knowledge related to the tool."
    else:
        # Fallback message if API response is invalid
        if language == "zh":
            comment_body = "🤖 基于AI机器人的issue内容完整性检查结果:\n\n❌ 检查过程中发生错误，无法完成检查。请稍后重试或联系仓库管理员。\n\n👉 如果想重新检查，在评论区@issue_checker即可。"
        else:
            comment_body = "🤖 issue content check result from AI robot:\n\n❌ An error occurred during the check. Please try again later or contact the repository administrator.\n\n👉 If you want to re-check, please comment @issue_checker."

    # Post comment to GitHub issue
    repo = github.get_repo(repo_full_name)
    issue = repo.get_issue(number=issue_number)
    issue.create_comment(comment_body)

    # Extract check status from AI response
    if response is not None and hasattr(response, 'text'):
        # Print response text for debugging
        print(f"AI Response Text: {response.text}")
        # Patterns to match both Chinese and English status formats
        status_patterns = [
            r'\b(PASS|FAIL)\b',  # Match PASS or FAIL as whole words
            r'\[\s*(PASS|FAIL)\s*\]'  # Also match with optional brackets
        ]

        for pattern in status_patterns:
            match = re.search(pattern, response.text, re.IGNORECASE)
            if match:
                print(f"Match found: {match.group(0)}")
                check_status = match.group(1).upper()
                break
    print(f"check status: {check_status}")
    # Manage labels based on check status
    if check_status in ['PASS', 'FAIL']:
        # Get current labels
        current_labels = [label.name for label in issue.labels]

        # Define labels
        pass_label = 'content_check_passed'
        fail_label = 'content_check_failed'

        # Remove conflicting label if exists
        if check_status == 'PASS' and fail_label in current_labels:
            issue.remove_from_labels(fail_label)
        elif check_status == 'FAIL' and pass_label in current_labels:
            issue.remove_from_labels(pass_label)

        # Add the appropriate label
        if check_status == 'PASS' and pass_label not in current_labels:
            issue.add_to_labels(pass_label)
        elif check_status == 'FAIL' and fail_label not in current_labels:
            issue.add_to_labels(fail_label)

    print("Issue content check completed and comment posted.")

except Exception as e:
    print(f"Error occurred: {str(e)}")

    # Post error comment
    repo = github.get_repo(repo_full_name)
    issue = repo.get_issue(number=issue_number)
    if language == "zh":
        error_comment = f"""
        ## 问题内容检查失败

        在检查问题内容时发生错误：
        ```
        {str(e)}
        ```

        请稍后重试(评论区@issue_checker)或联系仓库管理员。
        """
    else:
        error_comment = f"""
        ## Issue Content Check Failed

        An error occurred while checking the issue content:
        ```
        {str(e)}
        ```

        Please try again later(comment @issue_checker) or contact the repository administrator.
        """
    issue.create_comment(error_comment)
