# 服务契约

采样和打分是两个端点。采样响应可以直接带采样时 logprob（`logprobs=true`），teacher-forced 打分仍走 `/generate`。

## 采样

`POST /v1/chat/completions`

请求里有 `seed`、`temperature`、`top_p`、`stream=false`、`reasoning_effort=xhigh`，以及：

```json
"chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
```

响应要有：

- `choices[0].finish_reason`
- `choices[0].message.content`
- `choices[0].message.reasoning_content`
- `usage.completion_tokens`

一组 8 条。前 4 条 `temperature=0.6`，后 4 条 `temperature=0.8`。`seed = seed_base + prompt_index * 100 + sample_index`。

`prompt_tokens + max_tokens + 16` 超过服务上下文时，客户端不发请求。

`prompt_tokens + completion_tokens` 超过 102400 时，这条记为删失，原文保留，不右截断。

### 采样 token id

服务带 `--rollout-score` 时，非流式请求可以加 `"return_token_ids": true`。响应顶层多两个字段：

- `prompt_token_ids`：模型实际输入的 prompt id（聊天模板渲染后）。
- `completion_token_ids`：按顺序提交的输出 id，含投机解码接受的 token 和最后的停止 token。长度等于 `usage.completion_tokens`。

打分时直接用 `prompt_token_ids + completion_token_ids`，`logprob_start_len = len(prompt_token_ids)`，不要把文本重新分词。没开 `--rollout-score` 或 `stream=true` 时返回 400。思考预算（`thinking_budget`）注入的控制 token 也在 `completion_token_ids` 里；rollout 不应设置思考预算。

### 采样时 logprob

非流式请求可以加 `"logprobs": true`（`top_logprobs` 只能是 0 或不填；`stream=true` 时返回 400）。响应的 `choices[0].logprobs` 形如：

```json
{"content": [{"token": "token_id:151667", "token_id": 151667, "logprob": -0.012,
              "sampling_logprob": -0.003, "kind": "accepted_draft",
              "bytes": null, "top_logprobs": []}],
 "logprob_mode": "raw", "sampling_logprob_mode": "processed", "recorded": true}
```

- `content` 与 `completion_token_ids` 一一对应，长度等于 `usage.completion_tokens`。
- `logprob`：发出该 token 那一轮的主模型 logits，全词表、温度 1 的 `log_softmax`，不做截断。与 `/generate` 打分是同一公式，是 trainer 要的口径。
- `sampling_logprob`：实际采样分布下的 logprob，即依次经过语法掩码、presence/frequency 惩罚、top-k（上限 20）、temperature、top-p、min-p 之后重新归一化的分布。贪心请求（`temperature=0`）恒为 0。在投机拒绝采样下，每个输出 token 的边缘分布正好是这个分布，所以可以直接当行为策略概率用于重要性采样比。开了 presence/frequency 惩罚时这个值是近似值；rollout 不应开惩罚。
- `kind`：`sampled`（首个 token、普通解码，或没有草稿的一轮）、`accepted_draft`（被接受的草稿 token）、`correction`（拒绝后从残差分布重采）、`bonus`（草稿全部接受后的额外 token）、`forced`（思考预算注入的控制 token，不是采样出来的，两个 logprob 都是 `null`）。
- 两个 logprob 都直接来自采样当轮的计算，不需要再打分。teacher-forced 的 `/generate` 打分可以用来交叉验证：同一份权重下两者只差数值噪声（采样走 verify 内核，打分走 prefill 内核）。

## 打分

`POST /generate`

```json
{
  "input_ids": [1, 2, 3, 4],
  "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
  "return_logprob": true,
  "logprob_start_len": 1,
  "top_logprobs_num": 0,
  "return_text_in_logprobs": false
}
```

响应：

```json
{"meta_info": {"input_token_logprobs": [[-0.1, 2], [-0.2, 3], [-0.3, 4]]}}
```

长度必须等于 `len(input_ids) - logprob_start_len`。每项是 `[logprob, token_id]` 或 `{"logprob": ..., "token_id": ...}`。`token_id` 必须等于 `input_ids[logprob_start_len + i]`。

这组数就是 trainer 要的 `old_logprobs`。它必须来自发出这些 token 的同一份权重。
