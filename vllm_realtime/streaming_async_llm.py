"""AsyncLLM extension for resumable prompt embeddings (tested on vLLM 0.19.1)."""

from __future__ import annotations

import asyncio
from typing import Any

from vllm import TokensPrompt
from vllm.v1.engine.async_llm import (
    AsyncLLM,
    InputStreamError,
    RequestOutputCollector,
    extract_prompt_components,
)


class StreamingEmbedsAsyncLLM(AsyncLLM):
    async def _add_streaming_input_request(
        self,
        request_id: str,
        input_stream,
        sampling_params,
        arrival_time=None,
        lora_request=None,
        tokenization_kwargs=None,
        trace_headers=None,
        priority: int = 0,
        data_parallel_rank=None,
    ) -> RequestOutputCollector:
        self._validate_streaming_input_sampling_params(sampling_params)
        inputs: dict[str, Any] = dict(
            supported_tasks=await self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
        )
        if not sampling_params.skip_clone:
            sampling_params = sampling_params.clone()
            sampling_params.skip_clone = True

        final_req = self.input_processor.process_inputs(
            request_id=request_id,
            prompt=TokensPrompt(prompt_token_ids=[0]),
            params=sampling_params,
            **inputs,
        )
        self.input_processor.assign_request_id(final_req)
        internal_req_id = final_req.request_id
        queue = RequestOutputCollector(
            sampling_params.output_kind, internal_req_id
        )

        async def handle_inputs() -> None:
            cancelled = False
            try:
                async for input_chunk in input_stream:
                    params = input_chunk.sampling_params or sampling_params
                    self._validate_streaming_input_sampling_params(params)
                    req = self.input_processor.process_inputs(
                        request_id=internal_req_id,
                        prompt=input_chunk.prompt,
                        params=params,
                        resumable=True,
                        **inputs,
                    )
                    req.external_req_id = request_id
                    prompt_text, _, _ = extract_prompt_components(
                        self.model_config, input_chunk.prompt
                    )
                    await self._add_request(
                        req, prompt_text, None, 0, queue
                    )
            except (asyncio.CancelledError, GeneratorExit):
                cancelled = True
            except Exception as error:  # noqa: BLE001
                queue.put(InputStreamError(error))
            finally:
                queue._input_stream_task = None
                if not cancelled:
                    await self._add_request(final_req, None, None, 0, queue)

        self._run_output_handler()
        queue._input_stream_task = asyncio.create_task(handle_inputs())
        return queue
