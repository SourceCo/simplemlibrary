package snickrs.ailibrary.autodiff.cnn;

import snickrs.ailibrary.autodiff.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;

public class Conv1DNode extends Node {
	//TODO: as of now this only supports the default configurations i provided
	public Conv1DConfig config;
	public Conv1DNode(Node left, Node right, Conv1DConfig config) {
		super(left, right);
		this.config = config;
	}
	public Conv1DNode(Node left, Node right) {
		super(left, right);
		this.config = new Conv1DConfig(right.shape()[0], 1, 0, 1, Conv1DConfig.NCW, PaddingMode.VALID);
	}
	public void recalculate_dims() {
		config.setK(children.get(1).shape()[0]);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.cnn().conv1d(children.get(0).evaluate(), children.get(1).evaluate(), this.config);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		// kernel --> [kS, iC, oC]
		// upstream --> [bS, oC, oW]
		// kernel * upstream = [kS, iC, oC] x [bS, oC, oW] --> [kS, iC, bS, oW]
		// permute to [bS, iC, kS, oW]
		// col2im(kernel * upstream) = [bS, iC, iW]
		partials.set(0, col2im1d(Nd4j.tensorMmul(children.get(1).m, upstream, new int[][] {{2}, {1}}).permute(2, 1, 0, 3), config.getS(), config.getP(), children.get(0).m.size(2), config.getS()));
		// input --> [bS, iC, iW]
		// im2col(input) --> [bS, iC, kS, oW]
		// upstream --> [bS, oC, oW]
		// gradient = im2col(input) * upstream --> [bS, iC, kS, oW] x [bS, oC, oW] --> [iC, kS, oC]
		// permute to [kS, iC, oC]
		partials.set(1, Nd4j.tensorMmul(im2col1d(children.get(0).m, config.getK(), config.getS(), config.getP(), config.getD()), upstream, new int[][] {{0, 3}, {0, 2}}).permute(1, 0, 2));
	}
    public INDArray col2im1d(INDArray input, long s, long p, long iW, long d) {
    	long[] i_shape = input.shape();
        long bS = i_shape[0];
        long iC = i_shape[1];
        long kS = i_shape[2];
        long oW = i_shape[3];
        INDArray output = Nd4j.zeros(bS, iC, iW);
        for (long n = 0; n < bS; n++) {
            for (long c = 0; c < iC; c++) {
                for (long k = 0; k < kS; k++) {
                    long w_start = k * d - p;
                    for (long w = 0; w < oW; w++) {
                        long v = w_start + w * s;
                        if (v >= 0 && v < iW) {
                            double val = input.getDouble(n, c, k, w);
                            double cur = output.getDouble(n, c, v);
                            output.putScalar(new long[]{n, c, v}, cur + val);
                        }
                    }
                }
            }
        }
        return output;
    }
    public INDArray im2col1d(INDArray input, long kS, long s, long p, long d) {
    	long[] i_shape = input.shape();
        long bS = i_shape[0];
        long iC = i_shape[1];
        long iW = i_shape[2];
        long oW = (iW + 2 * p - d * (kS - 1) - 1) / s + 1;
        INDArray output = Nd4j.zeros(bS, iC, kS, oW);
        for (int n = 0; n < bS; n++) {
            for (int c = 0; c < iC; c++) {
                for (int k = 0; k < kS; k++) {
                	long w_start = k * d - p;
                    for (int w = 0; w < oW; w++) {
                        long v = w_start + w * s;
                        if (v >= 0 && v < iW) {
                            output.putScalar(new long[] {n, c, k, w}, output.getDouble(n, c, k, w)+input.getDouble(n, c, v));
                        }
                    }
                }
            }
        }
        return output;
    }
	public long[] shape() {
		long[] input_shape = children.get(0).shape();
		long[] kernel_shape = children.get(1).shape();
		long batch_size = input_shape[0];
		long len = input_shape[2];
		long ic = kernel_shape[1];
		long oc = kernel_shape[2];
		long p = config.getP();
		long d = config.getD();
		long k = config.getK();
		long s = config.getS();
        if(ic != input_shape[1]) throw new IllegalArgumentException("number of channels do not match, received " + ic + " vs " + input_shape[1]);
        long out = (len + 2 * p - d * (k - 1) - 1) / s + 1;
		return new long[] {batch_size, oc, out};
	}
}
