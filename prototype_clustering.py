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
import shutil

from torch.cuda.amp import GradScaler, autocast
from sklearn.decomposition import PCA
from PIL import Image
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline, Cache
from torch.nn.parallel import DataParallel

"""
TODO
1. It is still slower than needs be, we can speed it up with distributed training
2. Prototypes cause weird 2D PCA plots -- why?
3. Need to do grid search of hyperparams such as num of prototypes
4. Prototypes are not spreading out much and loss plateus
"""


TEXT_BATCH_SIZE = 512
TEXT_LEN = 128
LR = 1e-1
L2 = 1e-8  # to regularize Adam with weight_decay
MIN_LR = 1e-5
LR_STEP_RATE = 2
LATENT_SIZE = 2048
NUM_PROTOTYPES = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DIR = 'SlimPajama-627B'
LAMBDA = 0.  # separation loss



def main():

	model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	tiny_llama = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
	tiny_llama = DataParallel(tiny_llama)
	tokenizer = AutoTokenizer.from_pretrained(model_name)

	prototypes = nn.Parameter(torch.randn(NUM_PROTOTYPES, LATENT_SIZE, dtype=torch.bfloat16, device=DEVICE), requires_grad=True)

	criterion = nn.CrossEntropyLoss()  # Example loss function
	lr_decay_factor = 0.9  # Factor to decrease the learning rate
	optimizer = optim.Adam([prototypes], lr=LR, weight_decay=L2)

	# Directory containing the CSV files
	root_data_dir = DIR + '/train/'

	# Metrics
	clustering_loss_data = list()
	separation_loss_data = list()
	loss_data = list()

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
								
							loss_c = c_loss(z, prototypes)
							loss_s = s_loss(prototypes) * LAMBDA

						loss = loss_s + loss_c

						clustering_loss_data.append(loss_c.item())
						separation_loss_data.append(loss_s.item())
						loss.backward()
						optimizer.step()
						optimizer.zero_grad()

						count += 1

						plot_prototypes_and_data(z, prototypes, count)
						plot_prototypes(prototypes, count)
						plot_loss(clustering_loss_data, separation_loss_data)

					create_gif('plots/gif_figs/proto/', 'gifs/proto.gif')
					create_gif('plots/gif_figs/proto_z/', 'gifs/proto_z.gif')

					file_count += 1

					# Decrease learning rate every .zst files
					if file_count % LR_STEP_RATE == 0:
						for param_group in optimizer.param_groups:
							new_lr = param_group['lr'] * lr_decay_factor
							param_group['lr'] = max(new_lr, MIN_LR)  # Ensure lr doesn't drop below MIN_LR
						print(f"Decreased learning rate to {optimizer.param_groups[0]['lr']} after processing {file_count} files")

					print(
						  '\nClust Loss:', round(  sum(clustering_loss_data[-50:]) / len(clustering_loss_data[-50:])  , 2),
						  '\nSep Loss:', round(  sum(separation_loss_data[-50:]) / len(separation_loss_data[-50:])  , 2),
						  ' -- Iter:', count,
						  ' -- Dir:', file_path, 
						  )

					print("\nTime Taken:", time.time() - start_time)

					torch.save(prototypes, 'weights/prototypes_'+str(count)+'.pth')



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


def c_loss(z, prototypes):
	z = z.to(prototypes.device)    
	with torch.enable_grad():
		distances = torch.cdist(z, prototypes, p=2)  # Euclidean distance
		min_distances, _ = torch.min(distances, dim=1)  # Changed this line
		loss = torch.mean(min_distances)
	return loss


def s_loss(prototypes):
	"""
	Calculate the separation loss for prototypes to encourage them to be far apart.

	Args:
		prototypes (torch.Tensor): A tensor of shape (n, 2048) representing the prototype vectors.

	Returns:
		torch.Tensor: A scalar loss value that encourages prototypes to be far apart.
	"""
	# Compute pairwise distances between prototype vectors
	# prototypes: (n, 2048)
	# diff: (n, n, 2048) -> pairwise differences
	diff = prototypes.unsqueeze(1) - prototypes.unsqueeze(0)  # Create a matrix of differences
	distances = torch.norm(diff, dim=2)  # Compute the Euclidean distances
	
	# Create a mask to zero out the diagonal (self-distances)
	mask = 1 - torch.eye(prototypes.size(0), device=prototypes.device)
	
	# Apply mask to keep only the off-diagonal elements
	distances = distances * mask
	
	# Sum the distances and take the negative
	loss = -torch.sum(distances)
	
	return loss



def plot_loss(clustering_loss_data, separation_loss_data):
	plt.figure(figsize=(10, 6))
	
	# Plot clustering loss data
	plt.plot(range(1, len(clustering_loss_data) + 1), clustering_loss_data, 'b-', label='Clustering Loss')
	
	# Plot separation loss data
	plt.plot(range(1, len(separation_loss_data) + 1), separation_loss_data, 'g-', label='Separation Loss')

	# Adding a trend line for clustering loss data
	z_clustering = np.polyfit(range(1, len(clustering_loss_data) + 1), clustering_loss_data, 1)
	p_clustering = np.poly1d(z_clustering)
	plt.plot(range(1, len(clustering_loss_data) + 1), p_clustering(range(1, len(clustering_loss_data) + 1)), "r--", alpha=0.8, label='Clustering Trend Line')
	
	# Adding a trend line for separation loss data
	z_separation = np.polyfit(range(1, len(separation_loss_data) + 1), separation_loss_data, 1)
	p_separation = np.poly1d(z_separation)
	plt.plot(range(1, len(separation_loss_data) + 1), p_separation(range(1, len(separation_loss_data) + 1)), "m--", alpha=0.8, label='Separation Trend Line')

	plt.title('Prototype Clustering and Separation Loss over Iterations')
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.grid(True)
	plt.legend()  # Add a legend to differentiate between the two lines
	
	plt.tight_layout()
	plt.savefig('plots/loss_plot.png')
	plt.close()

def plot_prototypes(prototypes, iteration, save_dir='plots/gif_figs/proto/'):

	pca = PCA(n_components=2)
	prototypes_2d = pca.fit_transform(prototypes.detach().cpu().float().numpy())

	# Create the plot
	plt.figure(figsize=(10, 8))
	plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], c='red', s=100, marker='*', label='Prototypes')
	
	# Add labels and title
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.title('2D Visualization of Prototypes')
	plt.legend()
	plt.grid(True)
	
	# Save the plot instead of showing it
	filename = os.path.join(save_dir, f'prototypes_and_data_iter_{iteration}.png')
	plt.savefig(filename)
	plt.close()  # Close the plot to free up memory
	
	
def plot_prototypes_and_data(z, prototypes, iteration, save_dir='plots/gif_figs/proto_z/'):
	
	# Combine z and prototypes for PCA
	z = z.to(prototypes.device)    
	combined = torch.cat([z, prototypes], dim=0)

	# Reduce dimensionality to 2D using PCA
	pca = PCA(n_components=2)
	combined_2d = pca.fit_transform(combined.detach().cpu().float().numpy())
	
	# Split back into z and prototypes
	z_2d = combined_2d[:z.shape[0]]
	prototypes_2d = combined_2d[z.shape[0]:]

	# Create the plot
	plt.figure(figsize=(12, 10))
	plt.scatter(z_2d[:, 0], z_2d[:, 1], c='blue', alpha=0.6, label='Data points')
	plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1], c='red', s=100, marker='*', label='Prototypes')
	
	# Add labels and title
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.title('2D Visualization of Data Points and Prototypes')
	plt.legend()
	plt.grid(True)
	
	# Save the plot instead of showing it
	filename = os.path.join(save_dir, f'prototypes_and_data_iter_{iteration}.png')
	plt.savefig(filename)
	plt.close()  # Close the plot to free up memory


def create_gif(image_folder, gif_name, duration=500):
	images = []
	for file_name in sorted(os.listdir(image_folder)):
		if file_name.endswith('.png'):
			file_path = os.path.join(image_folder, file_name)
			images.append(imageio.imread(file_path))
	imageio.mimsave(gif_name, images, duration=duration)


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


	
