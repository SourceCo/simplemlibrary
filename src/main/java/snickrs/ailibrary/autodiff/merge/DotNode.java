package snickrs.ailibrary.autodiff.merge;

import snickrs.ailibrary.autodiff.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DotNode extends Node {
	public int[][] axes;
	public DotNode(Node left, Node right, int[] l_axes, int[] r_axes) {
		super(left, right);
		this.axes = new int[][] {l_axes, r_axes};
	}
	public DotNode(Node left, Node right, int[][] axes) {
		super(left, right);
		this.axes = axes;
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.tensorMmul(children.get(0).evaluate(), children.get(1).evaluate(), axes);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		long[] l_ax = remove(range(children.get(0).m.shape()), axes[0]);
		long[] r_ax = remove(range(children.get(1).m.shape()), axes[1]);
		int[] left_axes = new int[l_ax.length];
		int[] right_axes = new int[r_ax.length];
		for(int i = 0; i < left_axes.length; i++) {
			left_axes[i] = i;
		}
		for(int i = 0; i < right_axes.length; i++) {
			right_axes[i] = i+l_ax.length;
		}
		INDArray left_grad = Nd4j.tensorMmul(upstream, children.get(1).m, new int[][] {right_axes, to_int(r_ax)});
		INDArray right_grad = Nd4j.tensorMmul(children.get(0).m, upstream, new int[][] {to_int(l_ax), left_axes});
		partials.set(0, left_grad.permute(permute_dims(left_grad.shape(), children.get(0).m.shape())));
		partials.set(1, right_grad.permute(permute_dims(right_grad.shape(), children.get(1).m.shape())));
	}
	public int[] permute_dims(long[] arr, long[] goal) {
		// ex: [30, 40, 20, 10, 50]
		// ex goal: [10, 20, 30, 40, 50]
		/// should return [3, 2, 0, 1, 4]
		int[] permutation = new int[goal.length];
		for(int i = 0; i < permutation.length; i++) {
			if(arr[i] == goal[i]) {
				permutation[i] = i;
				continue;
			}
			for(int j = 0; j < goal.length; j++) {
				if(arr[i] == goal[j]) {
					permutation[j] = i;
					break;
				}
			}
		}
		return permutation;
	}
	public long[] shape() {
		long[] l_shape = remove(children.get(0).shape(), axes[0]);
		long[] r_shape = remove(children.get(1).shape(), axes[1]);
		return concat(l_shape, r_shape);
	}
}
