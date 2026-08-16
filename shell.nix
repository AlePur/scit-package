{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    graphviz
    glib
    cairo
    libGL
    libGLU
    zlib
    stdenv.cc.cc
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.libGL
      pkgs.glib
      pkgs.cairo
      pkgs.zlib
    ]}:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    source ./venv/bin/activate
  '';
}
