package snickrs.ailibrary.autodiff.pooling;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Pooling3D.Pooling3DType;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import snickrs.ailibrary.autodiff.*;

public class AvgPooling3DNode extends UnaryNode {
	public INDArray mask;
	public Pooling3DConfig config;
	public AvgPooling3DNode(Node left, Pooling3DConfig config) {
		super(left);
		this.config = config;
	}
	public AvgPooling3DNode(Node left) {
		super(left);
		this.config = new Pooling3DConfig();
		config.setKD(2);
		config.setKH(2);
		config.setKW(2);
		config.setSD(config.getKD());
		config.setSH(config.getKH());
		config.setSW(config.getKW());
		config.setDD(1);
		config.setDH(1);
		config.setDW(1);
		config.setPD(0);
		config.setPH(0);
		config.setPW(0);
		config.setType(Pooling3DType.AVG);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.cnn().avgPooling3d(children.get(0).evaluate(), config);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
        long[] input_shape = children.get(0).m.shape();
        long[] shape = this.m.shape();
        long depth = input_shape[2];
        long height = input_shape[3];
        long width = input_shape[4];
        long kd = config.getKD();
        long kh = config.getKH();
        long kw = config.getKW();
        long sd = config.getSD();
        long sh = config.getSH();
        long sw = config.getSW();
        this.mask = Nd4j.zerosLike(children.get(0).m);
        for (int n = 0; n < shape[0]; n++) {
            for (int c = 0; c < shape[1]; c++) {
            	for(int d = 0; d < shape[2]; d++) {
            		for (int h = 0; h < shape[3]; h++) {
            			for (int w = 0; w < shape[4]; w++) {
            				long dStart = d * sd;
            				long hStart = h * sh;
            				long wStart = w * sw;
            				long dEnd = Math.min(dStart + kd, depth);
            				long hEnd = Math.min(hStart + kh, height);
            				long wEnd = Math.min(wStart + kw, width);
            				this.mask.put(new INDArrayIndex[] {
                                NDArrayIndex.point(n), NDArrayIndex.point(c), NDArrayIndex.interval(dStart, dEnd), 
                                NDArrayIndex.interval(hStart, hEnd), NDArrayIndex.interval(wStart, wEnd)
            				}, upstream.getDouble(n, c, d, h, w)/((dEnd-dStart)*(hEnd-hStart)*(wEnd-wStart)));
            			}
            		}
            	}
            }
        }
		partials.set(0, this.mask);
	}
    @Override
    public long[] shape() {
        long[] input_shape = children.get(0).shape();
        long oD = (input_shape[2] - config.getKD()) / config.getSD() + 1;
        long oH = (input_shape[3] - config.getKH()) / config.getSH() + 1;
        long oW = (input_shape[4] - config.getKW()) / config.getSW() + 1;
        return new long[] {input_shape[0], input_shape[1], oD, oH, oW};
    }
}
