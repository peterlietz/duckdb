#include "core_functions/scalar/struct_functions.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/expression/bound_expression.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/storage/statistics/struct_stats.hpp"
#include "duckdb/planner/expression_binder.hpp"

namespace duckdb {

static void StructDropFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &starting_vec = args.data[0];
	starting_vec.Verify(args.size());

	auto &starting_child_entries = StructVector::GetEntries(starting_vec);
	auto &result_child_entries = StructVector::GetEntries(result);

	auto &starting_types = StructType::GetChildTypes(starting_vec.GetType());

	auto &func_args = state.expr.Cast<BoundFunctionExpression>().children;
	auto fields_to_drop = case_insensitive_set_t();

	// Build a set of field names to drop from arguments 1 onwards
	for (idx_t arg_idx = 1; arg_idx < func_args.size(); arg_idx++) {
		auto &drop_field = func_args[arg_idx];

		// Extract field name from constant string expression
		if (drop_field->type == ExpressionType::VALUE_CONSTANT) {
			auto &const_expr = drop_field->Cast<BoundConstantExpression>();
			if (const_expr.value.type().id() == LogicalTypeId::VARCHAR) {
				string field_name = const_expr.value.GetValue<string>();
				fields_to_drop.insert(field_name);
			}
		}
	}

	// Copy only the fields that are NOT in the drop list
	idx_t result_idx = 0;
	for (idx_t field_idx = 0; field_idx < starting_child_entries.size(); field_idx++) {
		auto &starting_child = starting_child_entries[field_idx];
		auto drop = fields_to_drop.find(starting_types[field_idx].first.c_str());

		if (drop == fields_to_drop.end()) {
			// This field should be kept, copy it to result
			result_child_entries[result_idx++]->Reference(*starting_child);
		}
		// If found in drop list, skip it (don't add to result)
	}

	result.Verify(args.size());
	if (args.AllConstant()) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static unique_ptr<FunctionData> StructDropBind(ClientContext &context, ScalarFunction &bound_function,
                                               vector<unique_ptr<Expression>> &arguments) {
	if (arguments.empty()) {
		throw InvalidInputException("struct_drop: Missing required arguments");
	}
	if (LogicalTypeId::STRUCT != arguments[0]->return_type.id()) {
		throw InvalidInputException("struct_drop: First argument must be a STRUCT");
	}
	if (arguments.size() < 2) {
		throw InvalidInputException("struct_drop: Must specify at least one field name to drop");
	}

	child_list_t<LogicalType> new_children;
	auto &existing_children = StructType::GetChildTypes(arguments[0]->return_type);

	auto fields_to_drop = case_insensitive_set_t();

	// Validate incoming arguments (field names to drop) and record them
	for (idx_t arg_idx = 1; arg_idx < arguments.size(); arg_idx++) {
		auto &child = arguments[arg_idx];

		// Check for named parameter usage and reject it
		if (!child->GetAlias().empty()) {
			throw BinderException("struct_drop: Named parameters are not supported. Use string literals instead, e.g., "
			                      "struct_drop(struct, 'fieldname')");
		}

		// Only accept string constant arguments
		if (child->type != ExpressionType::VALUE_CONSTANT) {
			throw BinderException("struct_drop: Field names must be string literals (e.g., 'fieldname')");
		}

		auto &const_expr = child->Cast<BoundConstantExpression>();
		if (const_expr.value.type().id() != LogicalTypeId::VARCHAR) {
			throw BinderException("struct_drop: Field names must be string literals, not %s",
			                      const_expr.value.type().ToString().c_str());
		}

		string field_name = const_expr.value.GetValue<string>();

		if (fields_to_drop.find(field_name) != fields_to_drop.end()) {
			throw InvalidInputException("struct_drop: Duplicate field name '%s'", field_name.c_str());
		}
		fields_to_drop.insert(field_name);
	}

	// Keep only the fields that are NOT in the drop list
	for (idx_t field_idx = 0; field_idx < existing_children.size(); field_idx++) {
		auto &existing_child = existing_children[field_idx];
		auto drop = fields_to_drop.find(existing_child.first);
		if (drop == fields_to_drop.end()) {
			// This field is not being dropped, keep it
			new_children.push_back(make_pair(existing_child.first, existing_child.second));
		}
		// If found, skip it (drop this field)
	}

	// Check if any fields remain after dropping
	if (new_children.empty()) {
		throw InvalidInputException("struct_drop: Cannot drop all fields from a STRUCT");
	}

	bound_function.SetReturnType(LogicalType::STRUCT(new_children));
	return make_uniq<VariableReturnBindData>(bound_function.GetReturnType());
}

unique_ptr<BaseStatistics> StructDropStats(ClientContext &context, FunctionStatisticsInput &input) {
	auto &child_stats = input.child_stats;
	auto &expr = input.expr;

	auto fields_to_drop = case_insensitive_set_t();
	auto new_stats = StructStats::CreateUnknown(expr.return_type);

	// Build drop set from arguments
	for (idx_t arg_idx = 1; arg_idx < expr.children.size(); arg_idx++) {
		auto &drop_field = expr.children[arg_idx];

		if (drop_field->type == ExpressionType::VALUE_CONSTANT) {
			auto &const_expr = drop_field->Cast<BoundConstantExpression>();
			if (const_expr.value.type().id() == LogicalTypeId::VARCHAR) {
				string field_name = const_expr.value.GetValue<string>();
				fields_to_drop.insert(field_name);
			}
		}
	}

	auto existing_type = child_stats[0].GetType();
	auto existing_count = StructType::GetChildCount(existing_type);
	auto existing_stats = StructStats::GetChildStats(child_stats[0]);

	idx_t result_idx = 0;
	for (idx_t field_idx = 0; field_idx < existing_count; field_idx++) {
		auto &existing_child = existing_stats[field_idx];
		auto drop = fields_to_drop.find(StructType::GetChildName(existing_type, field_idx));
		if (drop == fields_to_drop.end()) {
			// Keep this field's statistics
			StructStats::SetChildStats(new_stats, result_idx++, existing_child);
		}
		// If found, skip (drop this field)
	}

	return new_stats.ToUnique();
}

ScalarFunction StructDropFun::GetFunction() {
	ScalarFunction fun({}, LogicalTypeId::STRUCT, StructDropFunction, StructDropBind, nullptr, StructDropStats);
	fun.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
	fun.varargs = LogicalType::ANY;
	fun.SetSerializeCallback(VariableReturnBindData::Serialize);
	fun.SetDeserializeCallback(VariableReturnBindData::Deserialize);
	return fun;
}

} // namespace duckdb
