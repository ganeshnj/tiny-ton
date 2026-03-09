// TinyTon operation definitions.

#ifndef TINYTON_OPS
#define TINYTON_OPS

include "tiny-ton/Dialect/TinyTon/TinyTonDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TinyTon_Op<string mnemonic, list<Trait> traits = []>
    : Op<TinyTon_Dialect, mnemonic, traits>;

// --- Placeholder operation to validate the build ---------------------------

def TinyTon_AddOp : TinyTon_Op<"add", [Pure, Commutative]> {
  let summary = "Integer addition.";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs I32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

#endif // TINYTON_OPS
