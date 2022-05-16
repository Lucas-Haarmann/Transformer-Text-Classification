from flask import Flask, render_template, request
from dataset_labels import idx_to_article
from inference import classifier_inference, longformer_inference

app = Flask(__name__)
@app.route("/")
def index():
    return "<p>Welcome</p>"

@app.route("/classify")
def classify():
    return render_template('home_template.html')

@app.route('/baseline')
def baseline():
    return render_template('classify_template.html')

@app.route('/baseline/result/', methods = ['POST', 'GET'])
def baseline_result():
    if request.method == 'GET':
        return f"Submit text at /classify"
    if request.method == 'POST':
        form_data = request.form
        output_class = classifier_inference(form_data['input'])
        return "CLASS:  " + idx_to_article[output_class]

@app.route('/longformer')
def longformer():
    return render_template('longformer_template.html')

@app.route('/longformer/result/', methods = ['POST', 'GET'])
def longformer_result():
    if request.method == 'GET':
        return f"Submit text at /classify"
    if request.method == 'POST':
        form_data = request.form
        output_class = longformer_inference(form_data['input'])
        return "CLASS:  " + idx_to_article[output_class]

if __name__ == "__main__":
    app.run(debug=True)