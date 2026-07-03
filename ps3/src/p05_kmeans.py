"""PS3 Problem 5: k-means image compression.

Centroids are learned on a small image and then used to recolour a large image,
compressing it to ``num_clusters`` colours.  Both steps are fully vectorised:
pixel-to-centroid assignment uses a single broadcasted distance computation.
"""
from __future__ import annotations
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


def init_centroids(num_clusters, image):
    """Pick ``num_clusters`` random pixels as initial RGB centroids."""
    h, w, c = image.shape
    flat = image.reshape(-1, c)
    idx = np.random.randint(flat.shape[0], size=num_clusters)
    return flat[idx].astype(np.float64)


def _assign(pixels, centroids):
    """Nearest-centroid index for every pixel (vectorised)."""
    dists = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def update_centroids(centroids, image, max_iter=50, print_every=10):
    """Run Lloyd's algorithm to convergence (or ``max_iter``)."""
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(np.float64)
    centroids = centroids.copy()

    for it in range(1, max_iter + 1):
        assign = _assign(pixels, centroids)
        moved = 0.0
        for k in range(centroids.shape[0]):
            members = pixels[assign == k]
            if members.shape[0] > 0:
                new = members.mean(axis=0)
                moved = max(moved, np.linalg.norm(new - centroids[k]))
                centroids[k] = new
        if it % print_every == 0:
            loss = np.mean(np.linalg.norm(pixels - centroids[assign], axis=1))
            print(f'[p05] k-means iter {it}: max centroid move={moved:.4f}, '
                  f'mean distortion={loss:.4f}')
        if moved < 1e-4:
            print(f'[p05] k-means converged after {it} iterations')
            break
    return centroids


def compress_image(image, centroids):
    """Replace every pixel with its nearest centroid colour."""
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(np.float64)
    assign = _assign(pixels, centroids)
    return centroids[assign].reshape(h, w, c).astype(np.uint8)


def main(args):
    os.makedirs('output', exist_ok=True)
    np.random.seed(229)

    small = np.copy(mpimg.imread(args.small_path))
    print(f'[p05] small image shape: {small.shape}')

    centroids = init_centroids(args.num_clusters, small)
    print('[p05] running k-means on the small image ...')
    centroids = update_centroids(centroids, small, args.max_iter, args.print_every)

    large = np.copy(mpimg.imread(args.large_path))
    large.setflags(write=1)
    print(f'[p05] large image shape: {large.shape}')

    compressed = compress_image(large, centroids)

    # Save originals + compressed and report the compression ratio.
    for name, img in [('orig_large', large), ('updated_large', compressed)]:
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        plt.title(name)
        plt.savefig(f'output/{name}.png', bbox_inches='tight')
        plt.close()

    bits_orig = 24  # 8 bits per RGB channel
    bits_comp = int(np.ceil(np.log2(args.num_clusters)))
    ratio = bits_orig / bits_comp
    print(f'[p05] {args.num_clusters} colours -> {bits_comp} bits/pixel '
          f'(vs {bits_orig}); compression ratio ~ {ratio:.1f}x')
    return dict(num_clusters=args.num_clusters, bits_per_pixel=bits_comp, ratio=ratio)


def build_args(data_dir='../../data/ps3'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default=f'{data_dir}/peppers-small.tiff')
    parser.add_argument('--large_path', default=f'{data_dir}/peppers-large.tiff')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--num_clusters', type=int, default=16)
    parser.add_argument('--print_every', type=int, default=10)
    return parser.parse_args([])


if __name__ == '__main__':
    main(build_args())
