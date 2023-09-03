from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def generate_keywords(content, pipe4keyword, max_gen_len=128):
    keywords = pipe4keyword("摘要生成:\n" + content, max_length=max_gen_len, do_sample=True)
    return keywords[0]['generated_text']


if __name__ == '__main__':
    # Set up Model, Tokenizer, Pipeline
    t5_model = AutoModelForSeq2SeqLM.from_pretrained('../model/t5-model')
    t5_tokenizer = AutoTokenizer.from_pretrained('../model/t5-tokenizer/')
    # If you have a nvidia GPU on your machine, please set device=0
    # pipe = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer, device=0)
    pipe = pipeline("text2text-generation", model=t5_model, tokenizer=t5_tokenizer)

    # Example
    test_content = ("城镇职工重大疾病保障水平对比分析——江苏省3个样本市资料统计分析。文章旨在分析不同经济发展水平下城镇职工重大疾病保障水平"
                    "存在的差异及影响因素.本研究在江苏省苏南、苏中、苏北各选取一个样本市,利用完全随机抽样方法分别于各样本市抽取300例重大疾"
                    "病患者资料,利用方差分析与Logistic回归模型,揭示了不同经济发展水平下城镇职工重大疾病保障情况,并提出建议. 城镇职工重大"
                    "疾病保障水平对比分析——江苏省3个样本市资料统计分析")
    test_keywords = "城镇职工重大疾病保障水平; 方差分析; Logistic回归; 对比分析"
    test_results = generate_keywords(test_content, pipe, max_gen_len=512)
    print("Real Keywords: ", test_keywords)
    print("Results: ", test_results)
