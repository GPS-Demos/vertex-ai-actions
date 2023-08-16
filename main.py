import json
import os
from flask import Response
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from icon import icon_data_uri
from utils import authenticate, handle_error, list_to_html, safe_cast, sanitize_and_load_json_str
from palm_api import model_with_limit_and_backoff, reduce, MODEL_TYPES, DEFAULT_MODEL_TYPE
from google.cloud import bigquery
from datetime import datetime

BASE_DOMAIN = 'https://{}-{}.cloudfunctions.net/{}-'.format(os.environ.get(
    'REGION'), os.environ.get('PROJECT'), os.environ.get('ACTION_NAME'))


# https://github.com/looker-open-source/actions/blob/master/docs/action_api.md#actions-list-endpoint
def action_list(request):
    """Return action hub list endpoint data for action"""
    auth = authenticate(request)
    if auth.status_code != 200:
        return auth

    response = {
        'label': 'Looker Vertex AI [DEV]',
        'integrations': [{
            'name': os.environ.get('ACTION_NAME'),
            'label': os.environ.get('ACTION_LABEL'),
            'supported_action_types': ['query'],
            "icon_data_uri": icon_data_uri,
            'form_url': BASE_DOMAIN + 'form',
            'url': BASE_DOMAIN + 'execute',
            'supported_formats': ['json'],
            'supported_formattings': ['formatted'],
            'supported_visualization_formattings': ['noapply'],
            'params': [
                {'name': 'email', 'label': 'Email',
                    'user_attribute_name': 'email', 'required': True},
                {'name': 'user_id', 'label': 'User ID',
                    'user_attribute_name': 'id', 'required': True}
            ]
        }]
    }

    print('returning integrations json')
    return Response(json.dumps(response), status=200, mimetype='application/json')


# https://github.com/looker-open-source/actions/blob/master/docs/action_api.md#action-form-endpoint
def action_form(request):
    """Return form endpoint data for action"""
    auth = authenticate(request)
    if auth.status_code != 200:
        return auth

    request_json = request.get_json()
    form_params = request_json['form_params']
    print(form_params)

    default_question = 'Can you summarize the following dataset in 10 bullet points?'
    if 'question' in form_params:
        default_question = form_params['question']

    default_row_or_all = 'all'
    if 'row_or_all' in form_params:
        default_row_or_all = form_params['row_or_all']

    default_params = 'yes'
    if 'default_params' in form_params:
        default_params = form_params['default_params']

    default_model_type = ''
    if 'model_type' in form_params:
        default_model_type = form_params['model_type']

    default_look_name = ''
    if 'look_name' in form_params:
        default_look_name = form_params['look_name']

    # step 1 - select a prompt
    response = [{
        'name': 'question',
        'label': 'Type your AI prompt',
        'description': 'Type your prompt to generate a model response.',
        'type': 'textarea',
        'required': True,
        "default":  default_question
    },
        {
        'name': 'row_or_all',
        'label': 'Run per row or all results?',
        'description': "Choose whether to run the model on all the results together, or, individually per row.",
        'type': 'select',
        'required': True,
        "default":  default_row_or_all,
        'options': [{'name': 'all', 'label': 'All Results'},
                    {'name': 'row', 'label': 'Per Row'}],
    },
        {
        'name': 'default_params',
        'label': 'Default Parameters?',
        'description': "Select 'no' to customize text model parameters.",
        'type': 'select',
        'required': True,
        "default":  default_params,
        'options': [{'name': 'yes', 'label': 'Yes'},
                    {'name': 'no', 'label': 'No'}],
        'interactive': True  # dynamic field for model specific options
    },
        {
        'name': 'look_name',
        'label': 'Type your Look Name',
        'description': 'Copy and enter the schedule name from above look title.',
        'type': 'textarea',
        'required': True,
        "default":  default_look_name
    }]

    # step 2 - optional - choose model type
    if ('default_params' in form_params and
            form_params['default_params'] == 'no'):
        response.extend([{
            'name': 'model_type',
            'label': 'Model Type',
            'type': 'select',
            'default': default_model_type,
            'options': [{'name': MODEL_TYPES['text-bison']['name'], 'label': MODEL_TYPES['text-bison']['label']},
                        {'name': MODEL_TYPES['code-bison']['name'], 'label': MODEL_TYPES['code-bison']['label']}],
            'interactive': True
        }
        ])

    # step 3a - optional - customize model params used by both models
    if ('default_params' in form_params and
            form_params['default_params'] == 'no' and
            'model_type' in form_params):
        response.extend([{
            'name': 'temperature',
            'label': 'Temperature',
            'description': 'The temperature is used for sampling during the response generation, which occurs when topP and topK are applied (Acceptable values = 0.0–1.0)',
            'type': 'text',
            'default': '0.2',
        },
            {
            'name': 'max_output_tokens',
            'label': 'Max Output Tokens',
            'description': 'Maximum number of tokens that can be generated in the response (Acceptable values = 1–1024 for Text Bison. Acceptable values = 1-2048 for Code Bison.)',
            'type': 'text',
            'default': '1024',
        }
        ])

    # step 3b - optional - customize model params used by text-bison
    if ('default_params' in form_params and
            form_params['default_params'] == 'no' and
            'model_type' in form_params and
            form_params['model_type'] == DEFAULT_MODEL_TYPE):
        response.extend([{
            'name': 'top_k',
            'label': 'Top-k',
            'description': 'Top-k changes how the model selects tokens for output. Specify a lower value for less random responses and a higher value for more random responses. (Acceptable values = 1-40)',
            'type': 'text',
            'default': '40',
        },
            {
            'name': 'top_p',
            'label': 'Top-p',
            'description': 'Top-p changes how the model selects tokens for output. Specify a lower value for less random responses and a higher value for more random responses. (Acceptable values = 0.0–1.0)',
            'type': 'text',
            'default': '0.8',
        }
        ])

    print('returning form json: {}'.format(json.dumps(response)))
    return Response(json.dumps(response), status=200, mimetype='application/json')


# https://github.com/looker-open-source/actions/blob/master/docs/action_api.md#action-execute-endpoint
def action_execute(request):
    """Generate a response from Generative AI Studio from a Looker action"""
    auth = authenticate(request)
    if auth.status_code != 200:
        return auth

    request_json = request.get_json()
    attachment = request_json['attachment']
    action_params = request_json['data']
    form_params = request_json['form_params']
    question = form_params['question']
    look_name = form_params['look_name']
    print('request_json'+str(request_json))
    print('attachment'+str(attachment))
    print('action_params'+str(action_params))
    print('form_params'+str(form_params))

    maximum_max_output_tokens = MODEL_TYPES[form_params['model_type']]['max_output_tokens'] if 'model_type' in form_params else 1024

    model_type = DEFAULT_MODEL_TYPE if 'model_type' not in form_params else MODEL_TYPES[form_params['model_type']]['name']
    temperature = 0.2 if 'temperature' not in form_params else safe_cast(
        form_params['temperature'], float, 0.0, 1.0, 0.2)
    max_output_tokens = 1024 if 'max_output_tokens' not in form_params else safe_cast(
        form_params['max_output_tokens'], int, 1, maximum_max_output_tokens, 1024)
    top_k = 40 if 'top_k' not in form_params else safe_cast(
        form_params['top_k'], int, 1, 40, 40)
    top_p = 0.8 if 'top_p' not in form_params else safe_cast(
        form_params['top_p'], float, 0.0, 1.0, 0.8)

    # placeholder for model error email response
    body = 'There was a problem running the model. Please try again with less data. '
    summary = ''
    bq_summary = ''
    row_chunks = 50  # mumber of rows to summarize together
    try:
        all_data = sanitize_and_load_json_str(
            attachment['data'])
        if form_params['row_or_all'] == 'row':
            row_chunks = 1  # run function on each row individually

        summary = model_with_limit_and_backoff(
            all_data, question, row_chunks, model_type, temperature, max_output_tokens, top_k, top_p)

        # if row, zip prompt_result with all_data and send html table
        if form_params['row_or_all'] == 'row':
            for i in range(len(all_data)):
                all_data[i]['prompt_result'] = summary[i]
            body = list_to_html(all_data)

        # if all, send summary on top of all_data
        if form_params['row_or_all'] == 'all':
            if len(summary) == 1:
                bq_summary = summary[0]
                body = 'Prompt Result:<br><strong>{}</strong><br><br><br>'.format(
                    summary[0].replace('\n', '<br>'))
            else:
                reduced_summary = reduce(
                    '\n'.join(summary), model_type, temperature, max_output_tokens, top_k, top_p)
                bq_summary = '\n'.join(summary)
                body = 'Final Prompt Result:<br><strong>{}</strong><br><br>'.format(
                    reduced_summary.replace('\n', '<br>'))
                body += '<br><br><strong>Batch Prompt Result:</strong><br>'
                body += '<br><br><strong>Batch Prompt Result:</strong><br>'.join(
                    summary).replace('\n', '<br>') + '<br><br><br>'

            body += list_to_html(all_data)

    except Exception as e:
        body += 'PaLM API Error: ' + e.message
        print(body)

    if body == '':
        body = 'No response from model. Try asking a more specific question.'

    # try:
    #     # todo - make email prettier
    #     message = Mail(
    #         from_email=os.environ.get('EMAIL_SENDER'),
    #         to_emails=action_params['email'],
    #         subject='Your GenAI Report from Looker',
    #         html_content=body
    #     )

    #     sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
    #     response = sg.send(message)
    #     print('Message status code: {}'.format(response.status_code))
    # except Exception as e:
    #     error = handle_error('SendGrid Error: ' + e.message, 400)
    #     return error
    try:
        bq_client = bigquery.Client()
        current_dateTime = str(datetime.now())
        table = bq_client.get_table("{}.{}.{}".format("transportation-platform-376719", "transportation_data_aiml", "genai_vertex-ai-looker-actions"))
        rows_to_insert = [{u"query_time": current_dateTime, u"question_genai": str(question), u"look_name": str(look_name), u"action_parameters_looker": str(action_params),u"form_parameters_genai": str(form_params),u"answer_genai": bq_summary}]
        response = bq_client.insert_rows_json(table, rows_to_insert)
        print('Insert into BQ has started')
        print("bq_summary"+bq_summary)
        print(response)
        if response == []:
            print('Insert into BQ status successful: {}'.format(response))
    except Exception as e:
        error = handle_error('Insert into BQ Error: ' + e.message, 400)
        return error

    return Response(response='BQ data insertion is successful',status=200, mimetype='application/json')
