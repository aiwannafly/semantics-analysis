from langchain.llms import HuggingFaceEndpoint

import data.tokens

mistral_llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=data.tokens.HUGGINGFACEHUB_API_TOKEN,
    task="text-generation",
    model_kwargs={
        "temperature": .5,
        "max_new_tokens": 200
    }
)


def prepare_prompt(prompt: str) -> str:
    prompt = f"""
    Твоя роль - извлечение технических терминов из текста. Тебе нужно вернуть только их названия.
    
    Пример:
    ```
    Текст: В качестве базы данных для хранения информации используется SQLite.
    Технические термины: {{'terms': ['база данных', 'SQLite']}}
    ```
    
    Текст: {prompt}
    
    Напиши твой один ответ ниже.
    
    Технические термины:"""
    return prompt


paragraphs = [
    'Используя данный подход, можно провести значительно более быстрое обучение в сравнении со стандартным Fine '
    'Tuning и получить более качественные результаты в сравнении с Prompt Engineering. Более того, для отдельных '
    'задач результаты Prompt Tuning даже превосходят результаты Fine Tuning.',

    'Для работы с нейронными сетями отлично подходит PyTorch.',

    'Итак, по набору входных токенов с использованием фиксированных параметров модели θ будет сформирована матрица '
    'векторизованных токенов. Затем, soft prompt P будет векторизован весами, получим Pe. Далее, '
    'получаем конкатенацию вышеуказанных матриц [Pe;Xe], которая пойдет на следующий после Embedding слой LLM.',

    'Данная техника во многом похожа на Prompt Engineering. Существенным отличием является то, что промпт в этом '
    'случае будет формироваться не человеком (hard-prompt), а искусственным интеллектом (soft-prompt).'
]

for p in paragraphs:
    llm_response = mistral_llm(prepare_prompt(p))

    print(f"[PARAGRAPH]: {p}")
    print(f"[TERMS]: {llm_response}")
    print()
