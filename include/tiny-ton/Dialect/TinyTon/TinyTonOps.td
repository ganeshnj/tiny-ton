// TinyTon operation definitions.

#ifndef TINYTON_OPS
#define TINYTON_OPS

include "tiny-ton/Dialect/TinyTon/TinyTonDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TinyTon_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyTon_Dialect, mnemonic, traits>;

// Scalar element type: integer or float (32-bit or 16-bit).
def TinyTon_ScalarType : AnyTypeOf<[I32, F32, F16]>;

// --- Arithmetic ------------------------------------------------------------

def TinyTon_AddOp : TinyTon_Op<"add",
    [Pure, Commutative, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Addition (integer or float).";
  let arguments = (ins TinyTon_ScalarType:$lhs, TinyTon_ScalarType:$rhs);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyTon_SubOp : TinyTon_Op<"sub",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Subtraction (integer or float).";
  let arguments = (ins TinyTon_ScalarType:$lhs, TinyTon_ScalarType:$rhs);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyTon_MulOp : TinyTon_Op<"mul",
    [Pure, Commutative, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Multiplication (integer or float).";
  let arguments = (ins TinyTon_ScalarType:$lhs, TinyTon_ScalarType:$rhs);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyTon_DivOp : TinyTon_Op<"div",
    [Pure, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Division (float only for now).";
  let arguments = (ins TinyTon_ScalarType:$lhs, TinyTon_ScalarType:$rhs);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Math intrinsics -------------------------------------------------------

def TinyTon_ExpOp : TinyTon_Op<"exp",
    [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Exponential (float).";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def TinyTon_LogOp : TinyTon_Op<"log",
    [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Natural logarithm (float).";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def TinyTon_SqrtOp : TinyTon_Op<"sqrt",
    [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Square root (float).";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def TinyTon_RsqrtOp : TinyTon_Op<"rsqrt",
    [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Reciprocal square root (float).";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def TinyTon_AbsOp : TinyTon_Op<"abs",
    [Pure, AllTypesMatch<["operand", "result"]>]> {
  let summary = "Absolute value (integer or float).";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def TinyTon_MaxOp : TinyTon_Op<"max",
    [Pure, Commutative, AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let summary = "Maximum (integer or float).";
  let arguments = (ins TinyTon_ScalarType:$lhs, TinyTon_ScalarType:$rhs);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Reductions ------------------------------------------------------------

def TinyTon_ReduceSumOp : TinyTon_Op<"reduce_sum",
    [AllTypesMatch<["operand", "result"]>]> {
  let summary = "Sum reduction across all threads in the block.";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

def TinyTon_ReduceMaxOp : TinyTon_Op<"reduce_max",
    [AllTypesMatch<["operand", "result"]>]> {
  let summary = "Max reduction across all threads in the block.";
  let arguments = (ins TinyTon_ScalarType:$operand);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// --- Comparison ------------------------------------------------------------

def TinyTon_CmpLtOp : TinyTon_Op<"cmp_lt",
    [Pure, AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "Less-than comparison. Returns i32 1/0.";
  let arguments = (ins TinyTon_ScalarType:$lhs, TinyTon_ScalarType:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

// --- Constants -------------------------------------------------------------

def TinyTon_ConstOp : TinyTon_Op<"const", [Pure]> {
  let summary = "Materialize an integer immediate value.";
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$value attr-dict";
}

def TinyTon_FConstOp : TinyTon_Op<"fconst", [Pure]> {
  let summary = "Materialize a float32 immediate value.";
  let arguments = (ins F32Attr:$value);
  let results = (outs F32:$result);
  let assemblyFormat = "$value attr-dict";
}

def TinyTon_HConstOp : TinyTon_Op<"hconst", [Pure]> {
  let summary = "Materialize a float16 immediate value.";
  let arguments = (ins F32Attr:$value);
  let results = (outs F16:$result);
  let assemblyFormat = "$value attr-dict";
}

// --- Kernel arguments ------------------------------------------------------

// elem_type: 0 = i32, 1 = f32, 2 = f16 (matches ElementType enum)
def TinyTon_ArgOp : TinyTon_Op<"arg", [Pure]> {
  let summary = "Load a kernel argument by index.";
  let arguments = (ins I32Attr:$index,
                       DefaultValuedAttr<BoolAttr, "false">:$is_pointer,
                       DefaultValuedAttr<I32Attr, "0">:$elem_type);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$index (`ptr` $is_pointer^)? attr-dict `:` type($result)";
}

// --- Thread identification -------------------------------------------------

def TinyTon_ProgramIdOp : TinyTon_Op<"program_id", [Pure]> {
  let summary = "Returns the program (block) index along the given axis.";
  let arguments = (ins I32Attr:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$axis attr-dict";
}

def TinyTon_ThreadIdOp : TinyTon_Op<"thread_id", [Pure]> {
  let summary = "Returns the thread index within the block along the given axis.";
  let arguments = (ins I32Attr:$axis);
  let results = (outs I32:$result);
  let assemblyFormat = "$axis attr-dict";
}

// --- Memory ----------------------------------------------------------------

def TinyTon_LoadOp : TinyTon_Op<"load",
    [AttrSizedOperandSegments]> {
  let summary = "Load a value from data memory.";
  let arguments = (ins I32:$addr, Optional<I32>:$mask,
                       Optional<TinyTon_ScalarType>:$other);
  let results = (outs TinyTon_ScalarType:$result);
  let assemblyFormat = "$addr (`,` $mask^)? (`other` $other^ `:` type($other))? attr-dict `:` type($result)";
}

def TinyTon_StoreOp : TinyTon_Op<"store", []> {
  let summary = "Store a value to data memory.";
  let arguments = (ins I32:$addr, TinyTon_ScalarType:$value, Optional<I32>:$mask);
  let results = (outs);
  let assemblyFormat = "$addr `,` $value (`,` $mask^)? attr-dict `:` type($value)";
}

// --- Control flow ----------------------------------------------------------

def TinyTon_BranchZeroOp : TinyTon_Op<"bz", []> {
  let summary = "Skip 'skip' instructions if cond is zero.";
  let arguments = (ins I32:$cond, I32Attr:$skip);
  let results = (outs);
  let assemblyFormat = "$cond `,` $skip attr-dict";
}

def TinyTon_RetOp : TinyTon_Op<"ret", [Terminator]> {
  let summary = "Kernel return / thread done.";
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

#endif // TINYTON_OPS
