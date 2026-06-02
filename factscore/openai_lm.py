from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

PROJECT_ROOT = os.environ.get("LLM_UNCERTAINTY_ROOT", "/home/elp/project/llm_uncertainty")


def _load_project_env():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _first_env(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _configure_openai(key_path):
    _load_project_env()
    api_key = None
    if key_path and os.path.exists(key_path):
        with open(key_path, "r") as f:
            api_key = f.readline().strip()

    api_key = api_key or _first_env(
        "FACTSCORE_OPENAI_API_KEY",
        "CODEXAPIS_API_KEY",
        "OPENAI_API_KEY",
        "GPTGOD_API_KEY",
    )
    assert api_key, (
        "Please provide an API key via key_path or one of "
        "FACTSCORE_OPENAI_API_KEY, CODEXAPIS_API_KEY, OPENAI_API_KEY, GPTGOD_API_KEY."
    )
    openai.api_key = api_key

    api_base = _first_env(
        "FACTSCORE_OPENAI_BASE_URL",
        "CODEXAPIS_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "GPTGOD_BASE_URL",
    )
    if api_base:
        openai.api_base = api_base.rstrip("/")


def _chat_model_name(default="gpt-3.5-turbo"):
    return _first_env(
        "FACTSCORE_CHATGPT_MODEL",
        "FACTSCORE_OPENAI_MODEL",
        "CODEXAPIS_EXTRACTOR_MODEL",
        "OPENAI_MODEL",
    ) or default


def _completion_model_name(default="text-davinci-003"):
    return _first_env("FACTSCORE_INSTRUCTGPT_MODEL") or default


class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        _configure_openai(self.key_path)
        if self.model_name == "ChatGPT":
            self.model = _chat_model_name()
        elif self.model_name == "InstructGPT":
            self.model = _completion_model_name()
        else:
            self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, model_name=self.model, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            output = response["choices"][0]["message"]["content"]
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = call_GPT3(prompt, model_name=self.model, temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        else:
            raise NotImplementedError()

def call_ChatGPT(message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.ChatCompletion.create(model=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
            received = True
        except:
            # print(message)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                assert False
            
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(model=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
