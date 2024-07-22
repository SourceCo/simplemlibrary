package snickrs.ailibrary.autodiff.regularization;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import snickrs.ailibrary.autodiff.*;

public class ElasticNetPenaltyNode extends Node {
	public float coef; // this is the lambda coefficient
	public float alpha; // this sets the degree of mixing
	public ElasticNetPenaltyNode(Variable ... parameters) {
		super(parameters);
		this.coef = 0.001f;
		this.alpha = 0.1f;
	}
	public ElasticNetPenaltyNode(float coef, float alpha, Variable ... parameters) {
		super(parameters);
		this.coef = coef;
		this.alpha = alpha;
	}
	public ElasticNetPenaltyNode(List<Variable> parameters) {
		super(parameters);
		this.coef = 0.001f;
		this.alpha = 0.1f;
	}
	public ElasticNetPenaltyNode(float coef, float alpha, List<Variable> parameters) {
		super(parameters);
		this.coef = coef;
		this.alpha = alpha;
	}
	public INDArray child_evaluate() {
        float l1Sum = 0.0f;
        float l2Sum = 0.0f;
        for(int i = 0; i < children.size(); i++) {
            l1Sum += Transforms.abs(children.get(i).m).sumNumber().floatValue();
            l2Sum += Transforms.pow(children.get(i).m, 2.0f).sumNumber().floatValue();
        }
        this.m = Nd4j.scalar(coef * (alpha * l1Sum + (1.0f - alpha) * l2Sum));
        return this.m;
	}
	public void child_diff(INDArray upstream) {
		for(int i = 0; i < children.size(); i++) {
			INDArray p1 = Transforms.sign(children.get(i).m).mul(alpha);
			INDArray p2 = children.get(i).m.mul(2.0f*(1.0f-alpha));
			partials.set(i, unbroadcast(children.get(i).m, upstream.mul(p1.add(p2).mul(coef))));
		}
	}
}
