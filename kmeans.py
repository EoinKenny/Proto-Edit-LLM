import io
import os
import pandas as pd
import zstandard as zst
import json
import zstandard as zstd
import matplotlib.pyplot as plt
import time 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import imageio.v2 as imageio

from torch.cuda.amp import GradScaler, autocast
from sklearn.decomposition import PCA
from PIL import Image
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, Cache
from torch.nn.parallel import DataParallel
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors


TEXT_BATCH_SIZE = 1024
TEXT_LEN = 128
NUM_PROTOTYPES = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR = 'SlimPajama-627B'
LAMBDA = 1e-5  # separation loss


def main():

	model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	tiny_llama = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
	tiny_llama = DataParallel(tiny_llama)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	kmeans = MiniBatchKMeans(n_clusters=NUM_PROTOTYPES, batch_size=TEXT_BATCH_SIZE, n_init=3)

	# Directory containing the CSV files
	root_data_dir = DIR + '/train/'

	# Metrics
	inertia_data = []
	avg_min_distance_data = []
	avg_nn_distance_data = []


	count = 0
	file_count = 0
	start_time = time.time()

	# Iterate chunk directories
	for sub_dir in os.listdir(root_data_dir):
		sub_dir_path = os.path.join(root_data_dir, sub_dir)
		if os.path.isdir(sub_dir_path):  # Check if it's a directory

			# Iterate .zst files in chunk directories
			for zst_file in os.listdir(sub_dir_path):
				if zst_file.endswith('.zst'):
					file_path = os.path.join(sub_dir_path, zst_file)
					df = load_data(file_path)
					text_data = df.text.values.tolist()
					num_text_batches = len(text_data) // TEXT_BATCH_SIZE

					print(
						"DF Shape:", df.shape, 
						"  --len(text data):", len(text_data),
						"  --num text batches:", num_text_batches
						)

					file_count += 1

					# Here we start iterating the CSV in chunks
					for text_batch_idx in range(num_text_batches):
						text_batch_data = text_data[text_batch_idx * TEXT_BATCH_SIZE: (text_batch_idx+1) * TEXT_BATCH_SIZE]

						with torch.autocast(device_type=DEVICE):
							with torch.no_grad():
								input_ids = tokenizer(text_batch_data, max_length=TEXT_LEN, return_tensors="pt", padding=True, truncation=True).input_ids
								z = tiny_llama.module.model(input_ids).last_hidden_state
								z = z[:, -1, : ]
								
								z_numpy = z.detach().cpu().numpy()
								kmeans.partial_fit(z_numpy)
								cluster_centers = kmeans.cluster_centers_

						avg_min_distance = average_min_distance_to_centroids(z_numpy, cluster_centers)
						avg_min_distance_data.append(avg_min_distance)

						avg_nn_distance = average_nearest_neighbor_distance(z_numpy)
						avg_nn_distance_data.append(avg_nn_distance)

						print("NN Distance:", avg_nn_distance_data)



						inertia_data.append(kmeans.inertia_)

						count += 1

						plot_prototypes_and_data(z, cluster_centers, count)
						plot_prototypes(cluster_centers, count)
						plot_inertia(inertia_data)
						plot_avg_min_distance(avg_min_distance_data)

					create_gif('plots/gif_figs/proto/', 'gifs/proto.gif')
					create_gif('plots/gif_figs/proto_z/', 'gifs/proto_z.gif')

					file_count += 1

					print(
						  '\nInertia:', round(  sum(inertia_data[-50:]) / len(inertia_data[-50:])  , 2),
						  '\nClustering:', round(  sum(avg_min_distance_data[-50:]) / len(avg_min_distance_data[-50:])  , 2),
						  ' -- Iter:', count,
						  ' -- Dir:', file_path, 
						  )

					print("\nTime Taken:", time.time() - start_time)
					np.save(f'weights/cluster_centers_{count}.npy', kmeans.cluster_centers_)


def average_min_distance_to_centroids(data, centroids):
    distances = np.min(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
    return np.mean(distances)


def average_nearest_neighbor_distance(data):
    # Use 2 neighbors because the nearest neighbor of a point is itself
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean')
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    # Return the average of the distances to the second nearest neighbor (index 1)
    return np.mean(distances[:, 1])


def load_data(compressed_file_path) -> pd.DataFrame():
	def read_jsonl_zst(file_path) -> None:
		with open(file_path, 'rb') as file:
			decompressor = zst.ZstdDecompressor()
			stream_reader = decompressor.stream_reader(file)
			stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
			for line in stream:
				yield json.loads(line)
	data = list(read_jsonl_zst(compressed_file_path))
	df = pd.DataFrame(data)
	return df


def plot_prototypes(cluster_centers, iteration, save_dir='plots/gif_figs/proto/'):
    pca = PCA(n_components=2)
    centers_2d = pca.fit_transform(cluster_centers)
    plt.figure(figsize=(10, 8))
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=100, marker='*', label='Cluster Centers')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('2D Visualization of Cluster Centers')
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f'cluster_centers_iter_{iteration}.png')
    plt.savefig(filename)
    plt.close()


def plot_prototypes_and_data(z, cluster_centers, iteration, save_dir='plots/gif_figs/proto_z/'):
    combined = np.vstack([z.detach().cpu().numpy(), cluster_centers])
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    z_2d = combined_2d[:z.shape[0]]
    centers_2d = combined_2d[z.shape[0]:]
    plt.figure(figsize=(12, 10))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], c='blue', alpha=0.6, label='Data points')
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', s=100, marker='*', label='Cluster Centers')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('2D Visualization of Data Points and Cluster Centers')
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f'cluster_centers_and_data_iter_{iteration}.png')
    plt.savefig(filename)
    plt.close()


def plot_avg_min_distance(avg_min_distance_data):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_min_distance_data) + 1), avg_min_distance_data, 'g-', label='Avg Min Distance')
    # z = np.polyfit(range(1, len(avg_min_distance_data) + 1), avg_min_distance_data, 1)
    # p = np.poly1d(z)
    # plt.plot(range(1, len(avg_min_distance_data) + 1), p(range(1, len(avg_min_distance_data) + 1)), "r--", alpha=0.8, label='Trend Line')
    plt.title('Average Minimum Distance to Centroids over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Minimum Distance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/avg_min_distance_plot.png')
    plt.close()


def plot_inertia(inertia_data):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(inertia_data) + 1), inertia_data, 'b-', label='Inertia')
    z = np.polyfit(range(1, len(inertia_data) + 1), inertia_data, 1)
    p = np.poly1d(z)
    plt.plot(range(1, len(inertia_data) + 1), p(range(1, len(inertia_data) + 1)), "r--", alpha=0.8, label='Trend Line')
    plt.title('K-means Inertia over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/inertia_plot.png')
    plt.close()


def create_gif(image_folder, gif_name, duration=500):
	images = []
	for file_name in sorted(os.listdir(image_folder)):
		if file_name.endswith('.png'):
			file_path = os.path.join(image_folder, file_name)
			images.append(imageio.imread(file_path))
	imageio.mimsave(gif_name, images, duration=duration)


# Function to delete a directory and its contents
def delete_directory(path):
	if os.path.exists(path):
		try:
			shutil.rmtree(path)
			print(f"Deleted directory: {path}")
		except Exception as e:
			print(f"Error deleting directory {path}: {e}")

# Function to create a directory
def create_directory(path):
	try:
		os.makedirs(path, exist_ok=True)
		print(f"Created directory: {path}")
	except Exception as e:
		print(f"Error creating directory {path}: {e}")


if __name__ == '__main__':
	
	directories = [
		'gifs',
		'plots',
		'gif_figs',
		'gif_figs/proto',
		'gif_figs/proto_z',
		'plots/gif_figs/proto_z/',
		'plots/gif_figs/proto/',
	]

	for directory in directories:
		delete_directory(directory)
		create_directory(directory)

	main()


	
