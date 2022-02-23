import dash
import dash.dependencies as dd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from io import BytesIO
import base64
import pandas as pd
from collections import Counter
import warnings
import joblib

warnings.filterwarnings('ignore')
from wordcloud import WordCloud
import numpy as np
import dash_table
from collections import OrderedDict
import ld_and_sa

ld_sa = ld_and_sa.LanguageDetect_and_SentimentAnalysis()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                title='Language & Sentiment ', )  # external_stylesheets=external_stylesheets

data_clean = pd.read_csv('clean_language.csv')
data_l = OrderedDict([
    ('№', list(np.arange(17) + 1)),
    ("Lang", ['English', 'Malayalam', 'Hindi', 'Tamil', 'Portugeese', 'French', 'Dutch', 'Spanish', 'Greek', 'Russian',
              'Danish', 'Italian', 'Turkish', 'Sweedish', 'Arabic', 'German', 'Kannada']),
])

data_lang = pd.DataFrame(data_l)


def generate_lang_div(language):
    return html.Div(language, className='layered box')


app.layout = html.Div(
    id='particles-js',
    children=[
        html.H1('Language qualifier & Sentiment Analysis', id='logo'),
        html.Hr(),
        html.Div(
            id='main_up_part',
            children=[
                html.Div(
                    id="left_sight",
                    children=[
                        html.H5('Fill out the form in any language from the list', id='fill_text'),
                        dcc.Textarea(
                            id='textarea_input',
                            placeholder='Write the sentence...',
                        ),
                        html.Button('Submit', id='textarea_button', n_clicks=0),
                        html.Div(id='textarea_answer'),
                        html.Div(id='predict_lang'),
                        html.Div(id='final_ma_predict'),
                    ]
                ),
                html.Div(id='right_sight',
                         children=[generate_lang_div(i) for i in data_lang.Lang
                                   ]
                         )
            ]
        ),
        html.Div(
            children=[
                html.H1('Bonus!', id='bonus_text'),
                dcc.Dropdown(
                    options=[{'label': _, 'value': _} for _ in
                             ['English', 'French', 'Dutch', 'Spanish', 'Greek', 'Russian', 'Danish', 'Italian',
                              'Turkish', 'German']],
                    value='English',
                    id='dropdown_lang',
                ),
                html.Div(id='dd-output-container')
            ]
        ),
        html.Img(id="image_wc", style={'textAlign': 'center'}),
    ]
)


def plot_wordcloud(data):
    d = {a: x for a, x in data.values}
    wc = WordCloud(background_color='white', width=1000, height=560)
    wc.fit_words(d)
    return wc.to_image()


@app.callback(dd.Output('image_wc', 'src'),
              [dd.Input('image_wc', 'id')],
              Input('dropdown_lang', 'value')
              )
def make_image(b, lang):
    img = BytesIO()
    bags = Counter(
        np.concatenate(
            np.array([str(word).split(' ') for word in data_clean.Text[data_clean.Language == lang]], dtype=object),
            axis=0))
    word, freq = [], []
    for w, f in bags.most_common():
        word.append(w)
        freq.append(f)
    dfm = pd.DataFrame({'word': word, 'freq': freq})
    plot_wordcloud(data=dfm).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


@app.callback(
    Output('predict_lang', 'children'),
    Input('textarea_button', 'n_clicks'),
    State('textarea_input', 'value')
)
def make_prediction(n_clicks, value):
    if n_clicks > 0:
        res = ld_sa.runer(value)
        return (html.P(['Your language is: ', html.A('{0}'.format(res[0]), style={'color': '#323232'}), html.Br(),
                        'Your sentiment Analysis is: ',
                        html.A('{0}'.format("Positive" if res[1][0] == 1 else "Negative"), id="ma_color")]))
        # return 'Your language is: ' + res[0]


@app.callback(
    Output('ma_color', 'style'),
    [Input(component_id='ma_color', component_property='children')]
)
def change_style(input1):
    if input1 == "Positive":
        return {'color': '#33A853'}
    else:
        return {'color': '#EA4336'}


@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown_lang', 'value'),
)
def update_output(value):
    return 'You have selected {0}'.format(value)


@app.callback(
    Output('textarea_answer', 'children'),
    Input('textarea_button', 'n_clicks'),
    State('textarea_input', 'value')
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        return 'You have entered: \n' + value


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
