import tiktoken

# 正确转义后的完整文本
text = r'''prompt = f"""
            You are agent {agent_id} in a normal round (t >= 2) of a multi-agent debate system.

            You are given:
            - the original question
            - your own previous answer
            - structured historical information selected by the system

            You do NOT see other agents' new outputs from this same round.
            You may keep your previous answer or revise it.

            Return JSON only.
            Do not output markdown.
            Do not output any explanation outside JSON.

            Output JSON schema:
            {{
            "agent_id": "{agent_id}",
            "response_to_conflicts": [
                {{
                "conflict": "string",
                "response": "string",
                "status": "resolved|partially_resolved|still_open"
                }}
            ],
            "brief_reason": "string",
            "current_answer": "string"
            }}

            VERY IMPORTANT RULES:
            1. You must complete the fields in this order:
            (a) response_to_conflicts
            (b) brief_reason
            (c) current_answer
            2. current_answer is the FINAL answer for this round.
            3. If your reasoning changes anywhere in response_to_conflicts or brief_reason,
            you MUST update current_answer so that it matches your final view.
            4. current_answer is the single source of truth used by the system.
            5. Do NOT let current_answer disagree with response_to_conflicts or brief_reason.
            6. If you revise your answer during reasoning, the final revised answer must appear in current_answer.
            7. Keep response_to_conflicts concise and directly tied to the structured conflicts in the input.
            8. If there is no true unresolved conflict to respond to, response_to_conflicts may be [].

            Field instructions:
            - response_to_conflicts:
            Respond only to the unresolved conflicts represented in the structured input.
            Each item should contain:
            - conflict: the conflict text
            - response: your direct response to that conflict
            - status:
                resolved = fully addressed
                partially_resolved = some progress but not fully solved
                still_open = remains unresolved
            - brief_reason:
            A short summary of why you keep or revise your answer.
            - current_answer:
            The final answer after considering all conflict responses and reasoning above.

            Input:
            {json.dumps(payload, ensure_ascii=False, indent=2)}

            Return JSON only.
            """.strip()'''
print("123")
# GPT 系列通用编码
encoder = tiktoken.get_encoding("cl100k_base")
token_count = len(encoder.encode(text))

print(f"Token 总数：{token_count}")