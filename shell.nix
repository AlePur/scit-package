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
    cd $HOME/Skrivbord/pyproject/notebooks
    source ../venv/bin/activate
    export MYTOK="b6d8cb79d11d56f44849eab77a7e459837d165abf0c0767c"
    echo "$MYTOK" | xclip -selection clipboard
    jupyter notebook --no-browser --NotebookApp.token=$MYTOK
  '';
}
