from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams

import argparse
import json
import asyncio


class EEngine:
    def __init__(self, engine):
        self.engine = engine
        self.tokenizer = self.engine.engine.tokenizer

    async def generate_stream(
        self,
        prompt,
        request_id,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=4,
        stop_str=None,
        stop_token_ids=None,
        echo=False,
    ):
        context = prompt
        stop_token_ids = stop_token_ids or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        if isinstance(stop_str, str) and stop_str != "":
            stop = [stop_str]
        elif isinstance(stop_str, list) and stop_str != []:
            stop = stop_str
        else:
            stop = []

        for tid in stop_token_ids:
            stop.append(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            use_beam_search=False,
            stop=stop,
            max_tokens=max_new_tokens,
        )
        results_generator = engine.generate(context, sampling_params, request_id)

        async for request_output in results_generator:
            prompt = request_output.prompt
            if echo:
                text_outputs = [
                    prompt + output.text for output in request_output.outputs
                ]
            else:
                text_outputs = [output.text for output in request_output.outputs]
            text_outputs = request_output.outputs[-1].text
            # Note: usage is not supported yet
            ret = {"text": text_outputs, "error_code": 0, "usage": {}}
            await asyncio.sleep(2.1)
            yield text_outputs

    def sync_but_yield(
        self,
        prompt,
        request_id,
        temperature=1.0,
        top_p=1.0,
        max_new_tokens=4,
        stop_str=None,
        stop_token_ids=None,
        echo=True,
    ):
        loop = asyncio.get_event_loop()
        gen = self.generate_stream(
            prompt,
            request_id,
            temperature,
            top_p,
            max_new_tokens,
            stop_str,
            stop_token_ids,
            echo,
        )
        while True:
            try:
                value = loop.run_until_complete(gen.__anext__())
                yield value
            except StopAsyncIteration:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, default="distilgpt2")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )

    parser.add_argument("--num-gpus", type=int, default=1)

    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.model_path:
        args.model = args.model_path
    if args.num_gpus > 1:
        args.tensor_parallel_size = args.num_gpus

    engine_args = AsyncEngineArgs.from_cli_args(args)
    print(f"Engine args: {engine_args}")

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"Made einge")

    eengine = EEngine(engine)

    # async def run():
    #     async for v in eengine.generate_stream("User: Hello\nAssistant: ", 1):
    #         print(v)

    # import asyncio

    # asyncio.run(run())

    for v in eengine.sync_but_yield("User: Hello\nAssistant: ", 1):
        print(v)
