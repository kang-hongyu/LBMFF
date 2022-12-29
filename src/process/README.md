# 数据处理相关的代码
get_bert_sim.py 使用bert计算drug-drug、disease-disease相似度
get_drug_sim_all.py 使用smiles计算drug相似度，计算靶标相似度，计算副作用相似度，并进行加权融合
get_predict_result.py 找出预测结果中置信度最高的pair
merge_sim.py 将bert的结果和smiles的结果进行合并
