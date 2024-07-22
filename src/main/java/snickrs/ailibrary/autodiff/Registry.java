package snickrs.ailibrary.autodiff;

import snickrs.ailibrary.autodiff.activation.*;
import snickrs.ailibrary.autodiff.loss.*;
import snickrs.ailibrary.autodiff.dropout.*;
import snickrs.ailibrary.autodiff.math.*;
import snickrs.ailibrary.autodiff.regularization.ElasticNetPenaltyNode;
import snickrs.ailibrary.autodiff.regularization.L1PenaltyNode;
import snickrs.ailibrary.autodiff.regularization.L2PenaltyNode;

import java.util.HashMap;
import java.util.Map;

public class Registry {
    private static Map<Integer, Class<? extends Node>> registry = new HashMap<>();
    
    static {
    	// binary / stock nodes
        registry.put(0, AdditionNode.class);
        registry.put(1, ConstantNode.class);
        registry.put(2, ExponentNode.class);
        registry.put(3, GTNode.class);
        registry.put(4, L2NormNode.class);
        registry.put(5, LTNode.class);
        registry.put(6, MaxNode.class);
        registry.put(7, MeanNode.class);
        registry.put(8, MinNode.class);
        registry.put(9, MmulNode.class);
        registry.put(10, ModNode.class);
        registry.put(11, NegNode.class);
        registry.put(12, Node.class);
        registry.put(13, Placeholder.class);
        registry.put(14, ProductNode.class);
        registry.put(15, QuotientNode.class);
        registry.put(16, SubtractionNode.class);
        registry.put(17, SumNode.class);
        registry.put(18, UnaryNode.class);
        registry.put(19, Variable.class);
        registry.put(20, WhereNode.class);
        // activation
        registry.put(21, CeluNode.class);
        registry.put(22, CubeNode.class);
        registry.put(23, EluNode.class);
        registry.put(24, GCUNode.class);
        registry.put(25, GeluNode.class);
        registry.put(26, HardShrinkNode.class);
        registry.put(27, HardSigmoidNode.class);
        registry.put(28, HardSiluNode.class);
        registry.put(29, HardTanhNode.class);
        registry.put(30, LogSigmoidNode.class);
        registry.put(31, LogSoftmaxNode.class);
        registry.put(32, LReluNode.class);
        registry.put(33, MishNode.class);
        registry.put(34, RectifiedTanhNode.class);
        registry.put(35, Relu6Node.class);
        registry.put(36, ReluNode.class);
        registry.put(37, RReluNode.class);
        registry.put(38, SeluNode.class);
        registry.put(39, SigmoidNode.class);
        registry.put(40, SiluNode.class);
        registry.put(41, SoftmaxNode.class);
        registry.put(42, SoftminNode.class);
        registry.put(43, SoftplusNode.class);
        registry.put(44, SoftShrinkNode.class);
        registry.put(45, SoftsignNode.class);
        registry.put(46, SquareNode.class);
        registry.put(47, StepNode.class);
        registry.put(48, TanhShrinkNode.class);
        // loss
        registry.put(49, BCENode.class);
        registry.put(50, BFCENode.class);
        registry.put(51, CCENode.class);
        registry.put(52, CFCENode.class);
        registry.put(53, CosineDistanceNode.class);
        registry.put(54, ElasticNetPenaltyNode.class);
        registry.put(55, HingeNode.class);
        registry.put(56, HuberNode.class);
        registry.put(57, KLDivergenceNode.class);
        registry.put(58, L1PenaltyNode.class);
        registry.put(59, L2PenaltyNode.class);
        registry.put(60, LogCoshNode.class);
        registry.put(61, MAENode.class);
        registry.put(62, MAPENode.class);
        registry.put(63, MSENode.class);
        registry.put(64, MSLENode.class);
        registry.put(65, PoissonNode.class);
        registry.put(66, SquaredHingeNode.class);
        // math
        registry.put(67, AbsNode.class);
        registry.put(68, AcoshNode.class);
        registry.put(69, AcosNode.class);
        registry.put(70, AsinhNode.class);
        registry.put(71, AsinNode.class);
        registry.put(72, AtanhNode.class);
        registry.put(73, AtanNode.class);
        registry.put(74, CbrtNode.class);
        registry.put(75, CeilNode.class);
        registry.put(76, CoshNode.class);
        registry.put(77, CosNode.class);
        registry.put(78, ErfNode.class);
        registry.put(79, Expm1Node.class);
        registry.put(80, ExpNode.class);
        registry.put(81, FloorNode.class);
        registry.put(82, GammaNode.class);
        registry.put(83, Log1pNode.class);
        registry.put(84, LogNode.class);
        registry.put(85, RoundNode.class);
        registry.put(86, SignumNode.class);
        registry.put(87, SinhNode.class);
        registry.put(88, SinNode.class);
        registry.put(89, SqrtNode.class);
        registry.put(90, TanhNode.class);
        registry.put(91, TanNode.class);
        // dropout
        registry.put(92, AlphaDropoutNode.class);
        registry.put(93, DropoutNode.class);
        registry.put(94, GaussianDropoutNode.class);
        registry.put(95, GaussianNoiseNode.class);
        registry.put(96, SpatialDropoutNode.class);
        
        // i'm too lazy to sort
        registry.put(97, ClipNode.class);
    }

    public static Node createInstance(int code, Object ... args) {
    	try {
			return registry.get(code).getDeclaredConstructor().newInstance(args);
		} catch (Exception e) {
			throw new IllegalArgumentException("No class found for code: " + code);
		}
    }
}