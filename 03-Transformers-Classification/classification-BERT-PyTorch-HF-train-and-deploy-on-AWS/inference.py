'''
References:
https://workhuman.atlassian.net/wiki/spaces/whiqnlp/pages/9327519450/Packaging+Models+For+Deployment
https://github.com/aws-samples/aws-sagemaker-pytorch-shop-by-style/blob/master/src/similarity/inference.py
https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?highlight=model_fn#serve-a-pytorch-model
'''
import logging
import json
import torch
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()


CONTENT_TYPE = 'application/json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

model_type = 'albert-base-v2'
tokenizer  = AlbertTokenizer.from_pretrained( model_type, do_lower_case=True )


def model_fn(model_path='model_path'):

    dout  = 0.1
    model = AlbertForSequenceClassification.from_pretrained( model_type,
                                                             num_labels=2,
                                                             output_attentions=False,
                                                             output_hidden_states=False,
                                                             attention_probs_dropout_prob=dout,
                                                             hidden_dropout_prob=dout,
                                                            )
    model.to(DEVICE)
    if DEVICE == 'cuda':
        model.load_state_dict( torch.load( model_path ) )
    else:
        model.load_state_dict( torch.load( model_path, map_location=torch.device('cpu') ) )

    #model.eval()
    return model


def output_fn(prediction_output, accept=CONTENT_TYPE):
    logger.info('Serializing the generated output.')
    if accept == CONTENT_TYPE:
        return json.dumps(prediction_output)
    else:
        raise Exception('Requested unsupported ContentType in Accept: ' + accept)


def input_fn(request_body, request_content_type=CONTENT_TYPE):
    '''
        Load a pickled tensor
    '''
    if request_content_type == CONTENT_TYPE:
        request = json.loads(request_body)
        sentence = request['message']
        input_tokens = tokenizer(sentence)
        return input_tokens
    else:
        raise ValueError(f'Content type not supported {request_content_type}')


def predict_fn(input_data, model):

    logger.info('Making prediction')
    input_ids = torch.LongTensor(input_data['input_ids']).to(DEVICE)
    attn_mask = torch.LongTensor(input_data['attention_mask']).to(DEVICE)

    # incoming data should be of dimension (batch_size, sent_len)
    logger.info(f"data shape is: {input_ids.shape}")
    if len(input_ids.shape) == 1:
        input_ids = input_ids.view(1, -1)
        attn_mask = attn_mask.view(1, -1)
    logits = model(input_ids, attn_mask)['logits']
    predictions = (logits.argmax(-1)).long()
    return {"class_ids": predictions.tolist(),  # tensor objects not json serializable
            "confidences": logits.softmax(-1).tolist()}


if __name__ == '__main__':
    checkpoint_file = 'ckpts/albert/processed_ckpts_20220228T0754_albert_LR4e-5_bsize32_dout01_maxlen100_warmup1000_0.8546_0.8467_0.37/ckpts_20220228T0754/20220228T0754-epoch_7-val_loss_0.4031-f1micro_0.8429-f1macro0.8386.model'
    loaded_model = model_fn(checkpoint_file,)
    input_sentence = 'Way to go, guys!'
    request = {'message': input_sentence}
    request_string = json.dumps(request)              # to mirror the deployment
    input_tokens = input_fn(request_string)
    prediction_output = predict_fn( input_data=input_tokens,
                                    model=loaded_model, )
    output = output_fn(prediction_output)
    print(output)
