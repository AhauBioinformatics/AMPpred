BERT_model_INDEP = keras.models.load_model('EI dataset train 5fold/5fold/BERT-BFD_best_model.epoch03-loss0.19.hdf5')
X_test = load_INDEP_X_data('BERT_BFD')
BERT_mod_pred = BERT_model_INDEP.predict(X_test, batch_size=8)
print(BERT_mod_pred)
file = open('EI dataset train 5fold/5fold/result.txt', 'a')
for i in range(len(BERT_mod_pred)):
    # 去除[]
    mid = str(BERT_mod_pred[i]).replace('[', '').replace(']', '')
    # 删除单引号并用字符空格代替逗号
    mid = mid.replace("'", '').replace(',', '') + '\n'
    file.write(mid)
file.close()
BERT_mod_pred_labels = convert_preds(BERT_mod_pred)
print(BERT_mod_pred_labels)
file = open('EI dataset train 5fold/5fold/result(label).txt', 'a')
for i in range(len(BERT_mod_pred_labels)):
    # 去除[]
    mid = str(BERT_mod_pred_labels[i]).replace('[', '').replace(']', '')
    # 删除单引号并用字符空格代替逗号
    mid = mid.replace("'", '').replace(',', '') + '\n'
    file.write(mid)
file.close()

BERT_metrics = display_conf_matrix(y_test_INDEP, BERT_mod_pred_labels, BERT_mod_pred, 'BERT Model',
                                   'Validation/ADAPTABLE/BERT(LM)_Model_CM.png')
