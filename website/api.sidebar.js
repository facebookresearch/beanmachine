module.exports = [
'api/overview',
{
  type: 'category',
  label: 'beanmachine',
  collapsed: false,  items: [{
  type: 'category',
  label: 'beanmachine.applications',
  collapsed: true,  items: [{
  type: 'category',
  label: 'beanmachine.applications.clara',
  collapsed: true,  items: [{
  type: 'category',
  label: 'beanmachine.applications.clara.nmc',
  collapsed: true,  items: ["api/beanmachine.applications.clara.nmc", "api/beanmachine.applications.clara.nmc.State", "api/beanmachine.applications.clara.nmc.obtain_posterior", "api/beanmachine.applications.clara.nmc.simplex_proposer"],
}, {
  type: 'category',
  label: 'beanmachine.applications.clara.nmc_df',
  collapsed: true,  items: ["api/beanmachine.applications.clara.nmc_df", "api/beanmachine.applications.clara.nmc_df.LabelingErrorBMModel", "api/beanmachine.applications.clara.nmc_df.ModelOutput", "api/beanmachine.applications.clara.nmc_df.obtain_posterior"],
}],
}],
}, {
  type: 'category',
  label: 'beanmachine.ppl',
  collapsed: true,  items: ["api/beanmachine.ppl", "api/beanmachine.ppl.CompositionalInference", "api/beanmachine.ppl.GlobalHamiltonianMonteCarlo", "api/beanmachine.ppl.GlobalNoUTurnSampler", "api/beanmachine.ppl.Predictive", "api/beanmachine.ppl.RVIdentifier", "api/beanmachine.ppl.RejectionSampling", "api/beanmachine.ppl.SingleSiteAncestralMetropolisHastings", "api/beanmachine.ppl.SingleSiteHamiltonianMonteCarlo", "api/beanmachine.ppl.SingleSiteNewtonianMonteCarlo", "api/beanmachine.ppl.SingleSiteNoUTurnSampler", "api/beanmachine.ppl.SingleSiteRandomWalk", "api/beanmachine.ppl.SingleSiteUniformMetropolisHastings", "api/beanmachine.ppl.effective_sample_size", "api/beanmachine.ppl.empirical", "api/beanmachine.ppl.functional", "api/beanmachine.ppl.get_beanmachine_logger", "api/beanmachine.ppl.param", "api/beanmachine.ppl.r_hat", "api/beanmachine.ppl.random_variable", "api/beanmachine.ppl.simulate", "api/beanmachine.ppl.split_r_hat", {
  type: 'category',
  label: 'beanmachine.ppl.compiler',
  collapsed: true,  items: [{
  type: 'category',
  label: 'beanmachine.ppl.compiler.bmg_types',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.bmg_types", "api/beanmachine.ppl.compiler.bmg_types.AlwaysMatrix", "api/beanmachine.ppl.compiler.bmg_types.AnyRequirement", "api/beanmachine.ppl.compiler.bmg_types.BMGElementType", "api/beanmachine.ppl.compiler.bmg_types.BMGLatticeType", "api/beanmachine.ppl.compiler.bmg_types.BMGMatrixType", "api/beanmachine.ppl.compiler.bmg_types.BaseRequirement", "api/beanmachine.ppl.compiler.bmg_types.BooleanMatrix", "api/beanmachine.ppl.compiler.bmg_types.BroadcastMatrixType", "api/beanmachine.ppl.compiler.bmg_types.NaturalMatrix", "api/beanmachine.ppl.compiler.bmg_types.NegativeRealMatrix", "api/beanmachine.ppl.compiler.bmg_types.OneHotMatrix", "api/beanmachine.ppl.compiler.bmg_types.PositiveRealMatrix", "api/beanmachine.ppl.compiler.bmg_types.ProbabilityMatrix", "api/beanmachine.ppl.compiler.bmg_types.RealMatrix", "api/beanmachine.ppl.compiler.bmg_types.SimplexMatrix", "api/beanmachine.ppl.compiler.bmg_types.UpperBound", "api/beanmachine.ppl.compiler.bmg_types.ZeroMatrix", "api/beanmachine.ppl.compiler.bmg_types.abstractmethod", "api/beanmachine.ppl.compiler.bmg_types.always_matrix", "api/beanmachine.ppl.compiler.bmg_types.is_atomic", "api/beanmachine.ppl.compiler.bmg_types.is_convertible_to", "api/beanmachine.ppl.compiler.bmg_types.is_one", "api/beanmachine.ppl.compiler.bmg_types.is_zero", "api/beanmachine.ppl.compiler.bmg_types.lattice_to_bmg", "api/beanmachine.ppl.compiler.bmg_types.memoize", "api/beanmachine.ppl.compiler.bmg_types.must_be_matrix", "api/beanmachine.ppl.compiler.bmg_types.requirement_to_type", "api/beanmachine.ppl.compiler.bmg_types.supremum", "api/beanmachine.ppl.compiler.bmg_types.type_of_value", "api/beanmachine.ppl.compiler.bmg_types.upper_bound"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.compiler.hint',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.hint", "api/beanmachine.ppl.compiler.hint.log1mexp", "api/beanmachine.ppl.compiler.hint.math_log1mexp"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.compiler.patterns',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.patterns", "api/beanmachine.ppl.compiler.patterns.AnyPattern", "api/beanmachine.ppl.compiler.patterns.AtomicPattern", "api/beanmachine.ppl.compiler.patterns.AttributeSubpattern", "api/beanmachine.ppl.compiler.patterns.BoolPattern", "api/beanmachine.ppl.compiler.patterns.EmptyListPattern", "api/beanmachine.ppl.compiler.patterns.Fail", "api/beanmachine.ppl.compiler.patterns.FailPattern", "api/beanmachine.ppl.compiler.patterns.FloatPattern", "api/beanmachine.ppl.compiler.patterns.HeadTail", "api/beanmachine.ppl.compiler.patterns.IntPattern", "api/beanmachine.ppl.compiler.patterns.ListAll", "api/beanmachine.ppl.compiler.patterns.ListAny", "api/beanmachine.ppl.compiler.patterns.ListPattern", "api/beanmachine.ppl.compiler.patterns.MatchAny", "api/beanmachine.ppl.compiler.patterns.MatchEvery", "api/beanmachine.ppl.compiler.patterns.MatchResult", "api/beanmachine.ppl.compiler.patterns.NonePattern", "api/beanmachine.ppl.compiler.patterns.PatternBase", "api/beanmachine.ppl.compiler.patterns.PredicatePattern", "api/beanmachine.ppl.compiler.patterns.StringPattern", "api/beanmachine.ppl.compiler.patterns.Subpattern", "api/beanmachine.ppl.compiler.patterns.Success", "api/beanmachine.ppl.compiler.patterns.TypePattern", "api/beanmachine.ppl.compiler.patterns.abstractmethod", "api/beanmachine.ppl.compiler.patterns.attribute", "api/beanmachine.ppl.compiler.patterns.is_any", "api/beanmachine.ppl.compiler.patterns.match", "api/beanmachine.ppl.compiler.patterns.match_any", "api/beanmachine.ppl.compiler.patterns.match_every", "api/beanmachine.ppl.compiler.patterns.negate", "api/beanmachine.ppl.compiler.patterns.to_pattern", "api/beanmachine.ppl.compiler.patterns.type_and_attributes"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.compiler.performance_report',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.performance_report", "api/beanmachine.ppl.compiler.performance_report.PerformanceReport", "api/beanmachine.ppl.compiler.performance_report.event_list_to_report", "api/beanmachine.ppl.compiler.performance_report.json_to_perf_report"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.compiler.profiler',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.profiler", "api/beanmachine.ppl.compiler.profiler.Event", "api/beanmachine.ppl.compiler.profiler.ProfileReport", "api/beanmachine.ppl.compiler.profiler.ProfilerData", "api/beanmachine.ppl.compiler.profiler.event_list_to_report"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.compiler.rules',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.rules", "api/beanmachine.ppl.compiler.rules.AllChildren", "api/beanmachine.ppl.compiler.rules.AllListEditMembers", "api/beanmachine.ppl.compiler.rules.AllListMembers", "api/beanmachine.ppl.compiler.rules.AllOf", "api/beanmachine.ppl.compiler.rules.AllTermChildren", "api/beanmachine.ppl.compiler.rules.Check", "api/beanmachine.ppl.compiler.rules.Choose", "api/beanmachine.ppl.compiler.rules.Compose", "api/beanmachine.ppl.compiler.rules.Fail", "api/beanmachine.ppl.compiler.rules.FirstMatch", "api/beanmachine.ppl.compiler.rules.IgnoreException", "api/beanmachine.ppl.compiler.rules.ListEdit", "api/beanmachine.ppl.compiler.rules.OneChild", "api/beanmachine.ppl.compiler.rules.OneListMember", "api/beanmachine.ppl.compiler.rules.OrElse", "api/beanmachine.ppl.compiler.rules.PatternRule", "api/beanmachine.ppl.compiler.rules.Recursive", "api/beanmachine.ppl.compiler.rules.Rule", "api/beanmachine.ppl.compiler.rules.RuleDomain", "api/beanmachine.ppl.compiler.rules.RuleResult", "api/beanmachine.ppl.compiler.rules.SomeChildren", "api/beanmachine.ppl.compiler.rules.SomeListMembers", "api/beanmachine.ppl.compiler.rules.SomeOf", "api/beanmachine.ppl.compiler.rules.SpecificChild", "api/beanmachine.ppl.compiler.rules.Success", "api/beanmachine.ppl.compiler.rules.Trace", "api/beanmachine.ppl.compiler.rules.TryMany", "api/beanmachine.ppl.compiler.rules.TryOnce", "api/beanmachine.ppl.compiler.rules.abstractmethod", "api/beanmachine.ppl.compiler.rules.always_replace", "api/beanmachine.ppl.compiler.rules.at_least_once", "api/beanmachine.ppl.compiler.rules.either_or_both", "api/beanmachine.ppl.compiler.rules.if_then", "api/beanmachine.ppl.compiler.rules.ignore_div_zero", "api/beanmachine.ppl.compiler.rules.ignore_runtime_error", "api/beanmachine.ppl.compiler.rules.ignore_value_error", "api/beanmachine.ppl.compiler.rules.is_any", "api/beanmachine.ppl.compiler.rules.list_member_children", "api/beanmachine.ppl.compiler.rules.make_logger", "api/beanmachine.ppl.compiler.rules.match", "api/beanmachine.ppl.compiler.rules.pattern_rules", "api/beanmachine.ppl.compiler.rules.projection_rule", "api/beanmachine.ppl.compiler.rules.to_pattern"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.compiler.single_assignment',
  collapsed: true,  items: ["api/beanmachine.ppl.compiler.single_assignment", "api/beanmachine.ppl.compiler.single_assignment.SingleAssignment", "api/beanmachine.ppl.compiler.single_assignment.assign", "api/beanmachine.ppl.compiler.single_assignment.ast_boolop", "api/beanmachine.ppl.compiler.single_assignment.ast_compare", "api/beanmachine.ppl.compiler.single_assignment.ast_dict", "api/beanmachine.ppl.compiler.single_assignment.ast_dictComp", "api/beanmachine.ppl.compiler.single_assignment.ast_for", "api/beanmachine.ppl.compiler.single_assignment.ast_if", "api/beanmachine.ppl.compiler.single_assignment.ast_list", "api/beanmachine.ppl.compiler.single_assignment.ast_listComp", "api/beanmachine.ppl.compiler.single_assignment.ast_luple", "api/beanmachine.ppl.compiler.single_assignment.ast_return", "api/beanmachine.ppl.compiler.single_assignment.ast_setComp", "api/beanmachine.ppl.compiler.single_assignment.ast_while", "api/beanmachine.ppl.compiler.single_assignment.attribute", "api/beanmachine.ppl.compiler.single_assignment.aug_assign", "api/beanmachine.ppl.compiler.single_assignment.binop", "api/beanmachine.ppl.compiler.single_assignment.call", "api/beanmachine.ppl.compiler.single_assignment.expr", "api/beanmachine.ppl.compiler.single_assignment.index", "api/beanmachine.ppl.compiler.single_assignment.keyword", "api/beanmachine.ppl.compiler.single_assignment.match", "api/beanmachine.ppl.compiler.single_assignment.match_any", "api/beanmachine.ppl.compiler.single_assignment.match_every", "api/beanmachine.ppl.compiler.single_assignment.name", "api/beanmachine.ppl.compiler.single_assignment.negate", "api/beanmachine.ppl.compiler.single_assignment.single_assignment", "api/beanmachine.ppl.compiler.single_assignment.slice_pattern", "api/beanmachine.ppl.compiler.single_assignment.starred", "api/beanmachine.ppl.compiler.single_assignment.subscript", "api/beanmachine.ppl.compiler.single_assignment.unaryop"],
}],
}, {
  type: 'category',
  label: 'beanmachine.ppl.diagnostics',
  collapsed: true,  items: ["api/beanmachine.ppl.diagnostics", {
  type: 'category',
  label: 'beanmachine.ppl.diagnostics.common_plots',
  collapsed: true,  items: ["api/beanmachine.ppl.diagnostics.common_plots", "api/beanmachine.ppl.diagnostics.common_plots.SamplesSummary", "api/beanmachine.ppl.diagnostics.common_plots.autocorr", "api/beanmachine.ppl.diagnostics.common_plots.plot_helper", "api/beanmachine.ppl.diagnostics.common_plots.trace_helper", "api/beanmachine.ppl.diagnostics.common_plots.trace_plot"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.diagnostics.common_statistics',
  collapsed: true,  items: ["api/beanmachine.ppl.diagnostics.common_statistics", "api/beanmachine.ppl.diagnostics.common_statistics.confidence_interval", "api/beanmachine.ppl.diagnostics.common_statistics.effective_sample_size", "api/beanmachine.ppl.diagnostics.common_statistics.mean", "api/beanmachine.ppl.diagnostics.common_statistics.r_hat", "api/beanmachine.ppl.diagnostics.common_statistics.split_r_hat", "api/beanmachine.ppl.diagnostics.common_statistics.std"],
}, {
  type: 'category',
  label: 'beanmachine.ppl.diagnostics.diagnostics',
  collapsed: true,  items: ["api/beanmachine.ppl.diagnostics.diagnostics", "api/beanmachine.ppl.diagnostics.diagnostics.BaseDiagnostics", "api/beanmachine.ppl.diagnostics.diagnostics.Diagnostics", "api/beanmachine.ppl.diagnostics.diagnostics.make_subplots"],
}],
}],
}],
}
];
