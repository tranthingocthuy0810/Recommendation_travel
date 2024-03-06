from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load mô hình và dữ liệu
model = pickle.load(open("title_lk_model.pickle", "rb"))
dataset = pd.read_csv("data_tours.csv")


@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        tour_id = int(request.args.get('tour_id'))
    except ValueError:
        return jsonify({"error": "Invalid tour_id"}), 400

    recommendations = get_recommendations_from_model(tour_id)
    return jsonify({"recommendations": recommendations})


def get_recommendations_from_model(tour_id, threshold=0.1):
    sim = sorted(
        list(enumerate(model[tour_id])),
        key=lambda x: x[1],
        reverse=True,
    )
    index = [i[0] for i in sim if i[0] != tour_id and i[1] > threshold]

    cond1 = dataset.index.isin(index)
    cond2 = dataset.cluster == dataset.iloc[tour_id]['cluster']
    recommendations = dataset.loc[cond1 & cond2].sort_values(by='score', ascending=False).head(10)

    return recommendations[['title', 'description', 'sales_count', 'id', 'cluster']].to_dict(orient='records')


if __name__ == '__main__':
    app.run(debug=True)