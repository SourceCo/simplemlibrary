package snickrs.ailibrary.autodiff.reshape;

import org.nd4j.linalg.api.ndarray.INDArray;
import snickrs.ailibrary.autodiff.*;

public class PermuteNode extends UnaryNode {
    public int[] permute_dims;
    public int[] inverse_permute_dims;

    public PermuteNode(Node left, int ... permute_dims) {
        super(left);
        this.permute_dims = permute_dims;
        this.inverse_permute_dims = inverse(permute_dims);
    }

    public int[] inverse(int[] permutation) {
        int[] inverse = new int[permutation.length];
        for (int i = 0; i < permutation.length; i++) {
            inverse[permutation[i]] = i;
        }
        return inverse;
    }

    @Override
    public INDArray child_evaluate() {
        this.m = children.get(0).evaluate().permute(permute_dims);
        return this.m;
    }

    @Override
    public void child_diff(INDArray upstream) {
        partials.set(0, upstream.permute(inverse_permute_dims));
    }

    @Override
    public long[] shape() {
        long[] original_shape = children.get(0).shape();
        long[] new_shape = new long[original_shape.length];
        for (int i = 0; i < permute_dims.length; i++) {
            new_shape[i] = original_shape[permute_dims[i]];
        }
        return new_shape;
    }
}