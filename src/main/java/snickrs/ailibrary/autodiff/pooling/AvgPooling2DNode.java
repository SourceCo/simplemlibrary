package snickrs.ailibrary.autodiff.pooling;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling2D.Pooling2DType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import snickrs.ailibrary.autodiff.*;

public class AvgPooling2DNode extends UnaryNode {
	public INDArray mask;
	public Pooling2DConfig config;
	public AvgPooling2DNode(Node left, Pooling2DConfig config) {
		super(left);
		this.config = config;
	}
	public AvgPooling2DNode(Node left) {
		super(left);
		this.config = new Pooling2DConfig();
		config.setKH(2);
		config.setKW(2);
		config.setSH(config.getKH());
		config.setSW(config.getKW());
		config.setDH(1);
		config.setDW(1);
		config.setPH(0);
		config.setPW(0);
		config.setType(Pooling2DType.AVG);
		config.setPaddingMode(PaddingMode.VALID);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.cnn().avgPooling2d(children.get(0).evaluate(), config);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
        long[] input_shape = children.get(0).m.shape();
        long[] shape = this.m.shape();
        long height = input_shape[2];
        long width = input_shape[3];
        long kh = config.getKH();
        long kw = config.getKW();
        long sh = config.getSH();
        long sw = config.getSW();
        this.mask = Nd4j.zerosLike(children.get(0).m);
        for (int n = 0; n < shape[0]; n++) {
            for (int c = 0; c < shape[1]; c++) {
                for (int h = 0; h < shape[2]; h++) {
                    for (int w = 0; w < shape[3]; w++) {
                        long hStart = h * sh;
                        long wStart = w * sw;
                        long hEnd = Math.min(hStart + kh, height);
                        long wEnd = Math.min(wStart + kw, width);
                        this.mask.put(new INDArrayIndex[] {
                                NDArrayIndex.point(n), NDArrayIndex.point(c),
                                NDArrayIndex.interval(hStart, hEnd), NDArrayIndex.interval(wStart, wEnd)
                        }, upstream.getDouble(n, c, h, w)/((wEnd-wStart)*(hEnd-hStart)));
                    }
                }
            }
        }
		partials.set(0, this.mask);
	}
    @Override
    public long[] shape() {
        long[] input_shape = children.get(0).shape();
        long oH = (input_shape[2] - config.getKH()) / config.getSH() + 1;
        long oW = (input_shape[3] - config.getKW()) / config.getSW() + 1;
        return new long[] {input_shape[0], input_shape[1], oH, oW};
    }
}
