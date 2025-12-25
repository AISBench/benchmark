import json
import sys

def _get_generate_token_len(detail_json_path):
    data = []
    with open(detail_json_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    data.sort(key=lambda x: x["id"])
    generate_tokens = [item["output_tokens"] for item in data]
    return generate_tokens


if __name__ == "__main__":
    detail_json_path = sys.argv[1]
    generate_tokens_len = _get_generate_token_len(detail_json_path)
    if generate_tokens_len != [100, 200, 250, 111, 100, 200]:
        sys.exit(1)
    sys.exit(0)