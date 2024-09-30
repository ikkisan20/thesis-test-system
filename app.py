from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
import os

app = Flask(__name__)

def generate_random_data(num_samples):
    clothing_types = ['uniform and white t-shirt', 'formal shirt', 'colored shirt']
    data = {
        'id_length': np.random.randint(4, 16, size=num_samples),
        'entry_frequency': np.random.randint(1, 51, size=num_samples),
        'clothing_type': [random.choice(clothing_types) for _ in range(num_samples)]
    }
    return pd.DataFrame(data)

def apply_kmeans(df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['id_length', 'entry_frequency']])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    labels = {
        0: 'by-passer (unauthorized)',
        1: 'professor (authorized)',
        2: 'student (authorized)'
    }

    df['label'] = df['cluster'].map(labels)
    df['status'] = df.apply(lambda x: 'Warning' if x['label'] == 'by-passer (unauthorized)' else 'Authorized', axis=1)

    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_samples = int(request.form['num_samples'])
        df = generate_random_data(num_samples)
        df = apply_kmeans(df)

        # Save histogram as an image
        plt.figure(figsize=(10, 6))
        colors = {'student (authorized)': 'blue', 'professor (authorized)': 'green', 'by-passer (unauthorized)': 'red'}
        
        for label in df['label'].unique():
            subset = df[df['label'] == label]
            plt.hist(subset['entry_frequency'], bins=10, alpha=0.7, label=label, color=colors[label])
        
        plt.title('Entry Frequency Distribution by Group')
        plt.xlabel('Entry Frequency')
        plt.ylabel('Number of Individuals')
        plt.legend()
        plt.grid(axis='y')

        # Save the plot
        image_path = os.path.join('static', 'histogram.png')
        plt.savefig(image_path)
        plt.close()  # Close the plot to avoid display

        return render_template('index.html', tables=[df.to_html(classes='data')], image_path=image_path)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
