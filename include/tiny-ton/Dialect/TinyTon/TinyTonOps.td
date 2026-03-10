// TinyTon operation definitions.

#ifndef TINYTON_OPS
#define TINYTON_OPS

include "tiny-ton/Dialect/TinyTon/TinyTonDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TinyTon_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyTon_Dialect, mnemonic, traits>;

// --- Arithmetic ------------------------------------------------------------

def TinyTon_AddOp : TinyTon_Op<"add", [Pure, Commutative]> {
  let summary = "Integer addition.";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyTon_SubOp : TinyTon_Op<"sub", [Pure]> {
  let summary = "Integer subtraction.";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def TinyTon_MulOp : TinyTon_Op<"mul", [Pure, Commutative]> {
  let summary = "Integer multiplication.";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Comparison ------------------------------------------------------------

def TinyTon_CmpLtOp : TinyTon_Op<"cmp_lt", [Pure]> {
  let summary = "Integer less-than comparison. Returns 1 if lhs < rhs, else 0.";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// --- Constants -------------------------------------------------------------

def TinyTon_ConstOp : TinyTon_Op<"const", [Pure]> {
  let summary = "Materialize an immediate value.";
  let arguments = (ins I32Attr:$value);
  let results = (outs I32:$result);
  let assemblyFormat = "$value attr-dict";
}

// --- Kernel arguments ------------------------------------------------------

def TinyTon_ArgOp : TinyTon_Op<"arg", [Pure]> {
  let summary = "Load a kernel argument by index.";
  let arguments = (ins I32Attr:$index,
                       DefaultValuedAttr<BoolAttr, "false">:$is_pointer);
  let results = (outs I32:$result);
  let assemblyFormat = "$index (`ptr` $is_pointer^)? attr-dict";
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

def TinyTon_LoadOp : TinyTon_Op<"load", []> {
  let summary = "Load a value from data memory.";
  let arguments = (ins I32:$addr, Optional<I32>:$mask);
  let results = (outs I32:$result);
  let assemblyFormat = "$addr (`,` $mask^)? attr-dict `:` type($result)";
}

def TinyTon_StoreOp : TinyTon_Op<"store", []> {
  let summary = "Store a value to data memory.";
  let arguments = (ins I32:$addr, I32:$value, Optional<I32>:$mask);
  let results = (outs);
  let assemblyFormat = "$addr `,` $value (`,` $mask^)? attr-dict `:` type($addr)";
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
