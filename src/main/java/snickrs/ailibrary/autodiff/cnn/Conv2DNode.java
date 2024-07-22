package snickrs.ailibrary.autodiff.cnn;

import snickrs.ailibrary.autodiff.*;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;

public class Conv2DNode extends Node {
	//TODO: as of now this only supports the default configurations i provided
	public Conv2DConfig config;
	public Conv2DNode(Node left, Node right, Conv2DConfig config) {
		super(left, right);
		this.config = config;
	}
	public Conv2DNode(Node left, Node right) {
		super(left, right);
		this.config = new Conv2DConfig();
		long[] shape = right.shape();
		config.setSH(1);
		config.setSW(1);
		config.setDH(1);
		config.setDW(1);
		config.setKH(shape[0]);
		config.setKW(shape[1]);
		config.setPH(0);
		config.setPW(0);
		config.setPaddingMode(PaddingMode.VALID);
		config.setWeightsFormat(WeightsFormat.YXIO);
		config.setDataFormat(Conv2DConfig.NCHW);
		/**
		 * WeightsFormat reflects the possible formats for a convolution weight matrix.
		 * The following are available:
		 *   YXIO: [kH, kW, iC, oC]
		 *
		 *   OIYX:  [oC, iC, kH, kW]
		 *
		 *   OYXI: [oC, kH, kW, iC]
		 *
		 */
		/**
		 *   data formats
		 *   NCHW
		 *
		 *   NHWC
		 *
		 */
	}
	public void recalculate_dims() {
		long[] shape = children.get(1).shape();
		config.setKH(shape[0]);
		config.setKW(shape[1]);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.cnn().conv2d(children.get(0).evaluate(), children.get(1).evaluate(), this.config);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		long[] input_shape = children.get(0).m.shape();
		long iH = input_shape[2];
		long iW = input_shape[3];
		Col2Im op = new Col2Im(Nd4j.tensorMmul(children.get(1).m, upstream, new int[][] {{3}, {1}}).permute(3, 2, 0, 1, 4, 5), config);
		op.addIArgument(config.getSH());
        op.addIArgument(config.getSW());
        op.addIArgument(config.getPH());
        op.addIArgument(config.getPW());
        op.addIArgument(iH);
        op.addIArgument(iW);
        op.addIArgument(config.getDH());
        op.addIArgument(config.getDW());
        op.addIArgument(config.getPaddingMode().index);
		partials.set(0, Nd4j.exec(op)[0]);
		partials.set(1, Nd4j.tensorMmul(Nd4j.cnn().im2Col(children.get(0).m, config), upstream, new int[][] {{0, 4, 5}, {0, 2, 3}}).permute(1, 2, 0, 3));
//		partials.set(1, Nd4j.tensorMmul(children.get(1).m, upstream, new int[][] {{3}, {3}}));
//        long[] grad_shape = upstream.shape();
//        long[] input_shape = children.get(0).m.shape();
//        long kh = config.getKH();
//        long kw = config.getKW();
//        long sh = config.getSH();
//        long sw = config.getSW();
//        long ph = config.getPH();
//        long pw = config.getPW();
//        long dh = config.getDH();
//        long dw = config.getDW();
//		long h_factor = input_shape[2]-grad_shape[2];
//        long w_factor = input_shape[3]-grad_shape[3];
//        Conv2DConfig conf1 = Conv2DConfig.builder()
//                .kH(kh).kW(kw)
//                .sH(1).sW(1)
//                .dH(sh).dW(sw)
//                .pH(ph+h_factor).pW(pw+w_factor) // we want to do a full convolution, this is the workaround
//                .dataFormat(config.getDataFormat())
//                .weightsFormat(config.getWeightsFormat())
//                .paddingMode(PaddingMode.VALID)
//                .build();
//        // upstream: 1 x 5 x 7 x 7
//        // kernel: 4 x 4 x 3 x 5
//        // target shape: 1 x 3 x 10 x 10
//        partials.set(0, Nd4j.cnn().conv2d(upstream, Nd4j.reverse(children.get(1).m.dup()).permute(0, 1, 3, 2), conf1));
//        // Calculate gradient with respect to kernel
//        kh = grad_shape[2];
//        kw = grad_shape[3];
//        Conv2DConfig conf2 = Conv2DConfig.builder()
//                .kH(kh).kW(kw)
//                .sH(1).sW(1)
//                .dH(sh).dW(sw)
//                .pH(ph).pW(pw)
//                .dataFormat(config.getDataFormat())
//                .weightsFormat(config.getWeightsFormat())
//                .paddingMode(PaddingMode.VALID)
//                .build();
//        // input = 1 x 3 x 10 x 10 --> 3 x 1 x 10 x 10, dims 1, 0, 2, 3
//        // iC x oC x kH x kW
//        // 1 x 5 x 7 x 7 --> 7 x 7 x 1 x 5, dims 2, 3, 0, 1
//        partials.set(1, Nd4j.cnn().conv2d(children.get(0).m.permute(1, 0, 2, 3), upstream.permute(2, 3, 0, 1), conf2).permute(2, 3, 0, 1));
	}
	public long[] shape() {
		long[] input_shape = children.get(0).shape();
		long[] kernel_shape = children.get(1).shape();
		long batch_size = input_shape[0];
		long h = input_shape[2];
        long w = input_shape[3];
        long ic = kernel_shape[2];
        long oc = kernel_shape[3];
        long kh = kernel_shape[0];
        long kw = kernel_shape[1];
        long sh = config.getSH();
        long sw = config.getSW();
        long ph = config.getPH();
        long pw = config.getPW();
        long dh = config.getDH();
        long dw = config.getDW();
        if(ic != input_shape[1]) throw new IllegalArgumentException("number of channels do not match, received " + ic + " vs " + input_shape[1]);
        long output_height = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        long output_width = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
		return new long[] {batch_size, oc, output_height, output_width};
	}
}
