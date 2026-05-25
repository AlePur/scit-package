{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = [ pkgs.pandoc ];
  buildInputs = with pkgs; [
    graphviz
    glib
    libGL
    libGLU
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.libGL
      pkgs.glib
    ]}:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    source venv/bin/activate
    sphinx-build -M html docs/source/ docs/build/ -Ea
  '';
}
