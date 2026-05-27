{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    # Your existing dependencies
    graphviz
    glib
    cairo
    libGL
    libGLU
  ];
  #packages = [ pkgs.graphviz pkgs.opencv ];
  #LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/";
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.libGL
      pkgs.glib
      pkgs.cairo
    ]}:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    source ./venv/bin/activate
  '';
}
