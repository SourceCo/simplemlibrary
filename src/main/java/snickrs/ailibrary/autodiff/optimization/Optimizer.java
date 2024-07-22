package snickrs.ailibrary.autodiff.optimization;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import snickrs.ailibrary.autodiff.*;

public class Optimizer {
	public float lr; // by default 0.001
	public float clipnorm;
	public float clipvalue;
	public float global_clipnorm;
	public boolean use_ema;
	public float ema_momentum; // by default 0.99
	public int ema_overwrite_frequency;
	public float loss_scale_factor; 
	public int gradient_accumulation_steps;
	public float weight_decay; // by default 0.004
	public HashMap<Variable, INDArray> accumulated_gradients;
	public HashMap<Variable, INDArray> m_ema;
	public HashMap<Variable, INDArray> gradients;
	public Optimizer(float lr, float weight_decay) {
		this.lr = lr;
		this.weight_decay = weight_decay;
		this.accumulated_gradients = new HashMap<Variable, INDArray>();
		this.m_ema = new HashMap<Variable, INDArray>();
		this.gradients = new HashMap<Variable, INDArray>();
	}
	public INDArray expand(INDArray arr, int ... axis) {
		long[] shape = arr.shape().clone();
		INDArrayIndex[] indices = new INDArrayIndex[shape.length];
		for(int i = 0; i < shape.length; i++) {
			indices[i] = NDArrayIndex.interval(0, shape[i]);
		}
		for(int x : axis) {
			shape[x < 0 ? shape.length + x : x] += 1;
		}
		INDArray new_arr = Nd4j.zeros(shape);
		new_arr.put(indices, arr);
		return new_arr;
	}
	public void update(List<Variable> parameters) {
		for(int i = 0; i < parameters.size(); i++) {
			update(parameters.get(i));
		}
	}
	public void update(Variable variable) {
		if(gradients.get(variable) == null || !Arrays.equals(gradients.get(variable).shape(), variable.m.shape())) gradients.put(variable, variable.gradient);
		if(gradient_accumulation_steps != 0) {
			if(accumulated_gradients.get(variable) == null || !Arrays.equals(m_ema.get(variable).shape(), variable.m.shape())) accumulated_gradients.put(variable, Nd4j.zerosLike(variable.gradient));
			if(variable.updates % gradient_accumulation_steps != 0) {
				accumulated_gradients.get(variable).addi(variable.gradient.divi((float) gradient_accumulation_steps));
				return;
			} else {
				variable.gradient = accumulated_gradients.get(variable).dup();
			}
		}
		if(weight_decay != 0.0f) variable.gradient.muli(1.0f+weight_decay);
        if (clipnorm != 0.0f) {
            float norm = (float) variable.gradient.norm2Number();
            if (norm > clipnorm) {
                variable.gradient.muli(clipnorm / norm);
            }
        }
        if (clipvalue != 0.0f) {
            variable.gradient = Transforms.min(variable.gradient, clipvalue);
            variable.gradient = Transforms.max(variable.gradient, -clipvalue);
        }
        if(loss_scale_factor != 0.0f) {
        	variable.gradient.muli(1.0f/loss_scale_factor);
        }
        if (global_clipnorm != 0.0f) {
            float totalNorm = 0.0f;
            for (INDArray grad : gradients.values()) {
                totalNorm += (float) grad.norm2Number();
            }
            if (totalNorm > global_clipnorm) {
                for (INDArray grad : gradients.values()) {
                    grad.muli(global_clipnorm / totalNorm);
                }
            }
        }
        if (use_ema) {
        	if(m_ema.get(variable) == null || !Arrays.equals(m_ema.get(variable).shape(), variable.m.shape())) m_ema.put(variable, variable.m);
        	m_ema.get(variable).muli(ema_momentum).addi(variable.m.mul(1.0f-ema_momentum));
        }
        if (ema_overwrite_frequency != 0 && variable.updates % ema_overwrite_frequency == 0) {
        	variable.m = m_ema.get(variable).dup();
        }
		child_update(variable);
		if(gradient_accumulation_steps != 0 && variable.updates % gradient_accumulation_steps == 0) {
			accumulated_gradients.put(variable, Nd4j.zerosLike(variable.gradient));
		}
		variable.applyConstraints();
		variable.updates++;
	}
	public void child_update(Variable l) {
		
	}
}
