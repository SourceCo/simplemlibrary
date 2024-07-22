package snickrs.ailibrary.autodiff.pooling;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.Pooling2DType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import snickrs.ailibrary.autodiff.*;

public class AvgPooling1DNode extends UnaryNode {
	public INDArray mask;
	public Pooling2DConfig config;
	public AvgPooling1DNode(Node left, long k, long s, long p, long d, PaddingMode mode) {
		super(left);
		this.config = new Pooling2DConfig();
		config.setKH(1);
		config.setKW(k);
		config.setSH(1);
		config.setSW(s);
		config.setDH(1);
		config.setDW(d);
		config.setPH(0);
		config.setPW(p);
		config.setType(Pooling2DType.AVG);
		config.setPaddingMode(mode);
	}
	public AvgPooling1DNode(Node left) {
		super(left);
		this.config = new Pooling2DConfig();
		config.setKH(1);
		config.setKW(2);
		config.setSH(1);
		config.setSW(2);
		config.setDH(1);
		config.setDW(1);
		config.setPH(0);
		config.setPW(0);
		config.setType(Pooling2DType.AVG);
		config.setPaddingMode(PaddingMode.VALID);
	}
	public INDArray child_evaluate() {
		long[] shape = children.get(0).evaluate().shape();
		this.m = Nd4j.squeeze(Nd4j.cnn().avgPooling2d(children.get(0).m.reshape(shape[0], shape[1], 1, shape[2]), config), 2);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
        long[] input_shape = children.get(0).m.shape();
        long[] shape = this.m.shape();
        long width = input_shape[2];
        long kw = config.getKW();
        long sw = config.getSW();
        this.mask = Nd4j.zerosLike(children.get(0).m);
        for (int n = 0; n < shape[0]; n++) {
            for (int c = 0; c < shape[1]; c++) {
                    for (int w = 0; w < shape[2]; w++) {
                        long wStart = w * sw;
                        long wEnd = Math.min(wStart + kw, width);
                        this.mask.put(new INDArrayIndex[] {
                                NDArrayIndex.point(n), NDArrayIndex.point(c), NDArrayIndex.interval(wStart, wEnd)
                        }, upstream.getDouble(n, c, w)/(wEnd-wStart));
                    }
            }
        }
		partials.set(0, this.mask);
	}
    @Override
    public long[] shape() {
        long[] input_shape = children.get(0).shape();
        long oW = (input_shape[2] - config.getKW()) / config.getSW() + 1;
        return new long[] {input_shape[0], input_shape[1], oW};
    }
}
