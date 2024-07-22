package snickrs.ailibrary.autodiff.cnn;

import snickrs.ailibrary.autodiff.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;

public class Conv3DNode extends Node {
	//TODO: as of now this only supports the default configurations i provided
	public Conv3DConfig config;
	public Conv3DNode(Node left, Node right, Conv3DConfig config) {
		super(left, right);
		this.config = config;
	}
	public Conv3DNode(Node left, Node right) {
		super(left, right);
		this.config = new Conv3DConfig();
		long[] shape = right.shape();
		config.setSD(1);
		config.setSH(1);
		config.setSW(1);
		config.setDD(1);
		config.setDH(1);
		config.setDW(1);
		config.setKD(shape[0]);
		config.setKH(shape[1]);
		config.setKW(shape[2]);
		config.setPD(0);
		config.setPH(0);
		config.setPW(0);
		config.setPaddingMode(PaddingMode.VALID);
		config.setDataFormat(Conv3DConfig.NCDHW);
	}
	public void recalculate_dims() {
		long[] shape = children.get(1).shape();
		config.setKD(shape[0]);
		config.setKH(shape[1]);
		config.setKW(shape[2]);
	}
	public INDArray child_evaluate() {
		this.m = Nd4j.cnn().conv3d(children.get(0).evaluate(), children.get(1).evaluate(), this.config);
		return this.m;
	}
	public void child_diff(INDArray upstream) {
		// kernel: [kD, kH, kW, iC, oC]
		// upstream: [bS, oC, oD, oH, oW]
		// kernel * upstream = [kD, kH, kW, iC, oC] * [bS, oC, oD, oH, oW] --> [kD, kH, kW, iC, bS, oD, oH, oW]
		// permute to [bS, iC, kD, kH, kW, oD, oH, oW]
		// deconvolve to [bS, iC, iH, iD, iW]
		partials.set(0, deconv(Nd4j.tensorMmul(children.get(1).m, upstream, new int[][] {{4}, {1}}).permute(4, 3, 0, 1, 2, 5, 6, 7), config.getSD(), config.getSH(), config.getSW(), config.getPD(), config.getPH(), config.getPW(), children.get(0).m.size(2), children.get(0).m.size(3), children.get(0).m.size(4), config.getDD(), config.getDH(), config.getDW()));
		// input: [bS, iC, iD, iH, iW]
		// upstream: [bS, oC, oD, oH, oW]
		// convolve(input) = [bS, iC, kD, kH, kW, oD, oH, oW]
		// convolve(input) * upstream = [bS, iC, kD, kH, kW, oD, oH, oW] * [bS, oC, oD, oH, oW] = [iC, kD, kH, kW, oC]
		// permute to [kD, kH, kW, iC, oC]
		partials.set(1, Nd4j.tensorMmul(conv(children.get(0).m, config.getKD(), config.getKH(), config.getKW(), config.getSD(), config.getSH(), config.getSW(), config.getPD(), config.getPH(), config.getPW(), config.getDD(), config.getDH(), config.getDW()), upstream, new int[][] {{0, 5, 6, 7}, {0, 2, 3, 4}}).permute(1, 2, 3, 0, 4));
		System.out.println(partials.get(0).shapeInfoToString());
		System.out.println(partials.get(1).shapeInfoToString());
	}
	
	public INDArray deconv(INDArray input, long sD, long sH, long sW, long pD, long pH, long pW, long iD, long iH, long iW, long dD, long dH, long dW) {
		long[] i_shape = input.shape();
		long bS = i_shape[0];
        long iC = i_shape[1];
        long kD = i_shape[2];
        long kH = i_shape[3];
        long kW = i_shape[4];
        long oD = i_shape[5];
        long oH = i_shape[6];
        long oW = i_shape[7];
        INDArray output = Nd4j.zeros(bS, iC, iD, iH, iW);
        for(long n = 0; n < bS; n++) {
        	for(long c = 0; c < iC; c++) {
        		for(long d = 0; d < kD; d++) {
        			long d_start = d * dD - pD;
        			for(long h = 0; h < kH; h++) {
        				long h_start = h * dH - pH;
        				for(long w = 0; w < kW; w++) {
        					long w_start = w * dW - pW;
        					for(long cD = 0; cD < oD; cD++) {
        						long vD = d_start + cD * sD;
        						if(vD < 0 || vD >= iD) continue;
        						for(long cH = 0; cH < oH; cH++) {
        							long vH = h_start + cH * sH;
        							if(vH < 0 || vH >= iH) continue;
        							for(long cW = 0; cW < oW; cW++) {
        								long vW = w_start + cW * sW;
        								if(vW < 0 || vW >= iW) continue;
        								double val = input.getDouble(n, c, d, h, w, cD, cH, cW);
        								double cur = output.getDouble(n, c, vD, vH, vW);
        								output.putScalar(new long[]{n, c, vD, vH, vW}, cur+val);
        							}
        						}
        					}
        				}
        			}
        		}
        	}
        }		
        return output;
	}
	public INDArray conv(INDArray input, long kD, long kH, long kW, long sD, long sH, long sW, long pD, long pH, long pW, long dD, long dH, long dW) {
		long[] i_shape = input.shape();
		long bS = i_shape[0];
        long iC = i_shape[1];
        long iD = i_shape[2];
        long iH = i_shape[3];
        long iW = i_shape[4];
        long oD = (iD + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
        long oH = (iH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        long oW = (iW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
        INDArray output = Nd4j.zeros(bS, iC, kD, kH, kW, oD, oH, oW);
        for(long n = 0; n < bS; n++) {
        	for(long c = 0; c < iC; c++) {
        		for(long d = 0; d < kD; d++) {
        			long d_start = d * dD - pD;
        			for(long h = 0; h < kH; h++) {
        				long h_start = h * dH - pH;
        				for(long w = 0; w < kW; w++) {
        					long w_start = w * dW - pW;
        					for(long cD = 0; cD < oD; cD++) {
        						long vD = d_start + cD * sD;
        						if(vD < 0 || vD >= iD) continue;
        						for(long cH = 0; cH < oH; cH++) {
        							long vH = h_start + cH * sH;
        							if(vH < 0 || vH >= iH) continue;
        							for(long cW = 0; cW < oW; cW++) {
        								long vW = w_start + cW * sW;
        								if(vW < 0 || vW >= iW) continue;
        								double val = input.getDouble(n, c, vD, vH, vW);
        								double cur = output.getDouble(n, c, d, h, w, cD, cH, cW);
        								output.putScalar(new long[]{n, c, d, h, w, cD, cH, cW}, cur+val);
        							}
        						}
        					}
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
        INDArray output = Nd4j.create(bS, iC, kS, oW);
        for (int b = 0; b < bS; b++) {
            for (int c = 0; c < iC; c++) {
                for (int k = 0; k < kS; k++) {
                    for (int w = 0; w < oW; w++) {
                        long input_index = w * s - p + k * d;
                        if (input_index >= 0 && input_index < iW) {
                            output.putScalar(b, c, k, w, input.getDouble(b, c, input_index));
                        } else {
                            output.putScalar(b, c, k, w, 0.0d); // zero padding
                        }
                    }
                }
            }
        }
        return output;
    }
	public long[] shape() {
		long[] input_shape = children.get(0).shape(); // [bS, iC, iD, iH, iW]
		long[] kernel_shape = children.get(1).shape(); // [kD, kH, kW, iC, oC]
		long bS = input_shape[0];
		long iC = input_shape[1];
		long oC = kernel_shape[4];
        if(iC != kernel_shape[3]) throw new IllegalArgumentException("number of channels do not match, received " + iC + " vs " + kernel_shape[3]);
        
		long iD = input_shape[2];
		long iH = input_shape[3];
		long iW = input_shape[4];
		
		long pD = config.getPD();
		long pH = config.getPH();
		long pW = config.getPW();
		
		long dD = config.getDD();
		long dH = config.getDH();
		long dW = config.getDW();
		
		long kD = config.getKD();
		long kH = config.getKH();
		long kW = config.getKW();
		
		long sD = config.getSD();
		long sH = config.getSH();
		long sW = config.getSW();
		
        long oD = (iD + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
        long oH = (iH + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
        long oW = (iW + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
		return new long[] {bS, oC, oD, oH, oW};
	}
}
