{ pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    # Your existing dependencies
    pkgs.libgcc pkgs.zlib
  ];
  #packages = [ pkgs.graphviz pkgs.opencv ];
  #LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/";
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.libgcc pkgs.zlib (pkgs.python312.withPackages (ps: with ps; [
          numpy
          cython
          setuptools
          wheel
        ]))
    ]}:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    cd $HOME/Skrivbord/pyproject
    source venv/bin/activate
    ln -s /home/aleksander/Skrivbord/pyproject/venv/lib/python3.12/site-packages/numpy/_core/include /tmp/np-include
    python setup.py build_ext --inplace --include-dirs=/tmp/np-include
    #python setup.py build_ext --inplace --include-dirs=/home/aleksander/Skrivbord/pyproject/venv/lib/python3.12/site-packages/numpy/_core/include
  '';
}
