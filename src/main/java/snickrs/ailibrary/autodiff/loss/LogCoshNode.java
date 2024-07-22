package snickrs.ailibrary.autodiff.loss;

import snickrs.ailibrary.autodiff.*;
import snickrs.ailibrary.autodiff.math.*;

public class LogCoshNode extends UnaryNode {
	public LogCoshNode(Node target, Node prediction) {
		super(create(target, prediction));
	}
	public static Node create(Node target, Node prediction) {
		return new MeanNode(new LogNode(new CoshNode(new SubtractionNode(prediction, target))));
	}
}
