document.getElementById("exportBtn").addEventListener("click", function() {
    // Selecciona el elemento SVG
    const svg = document.getElementById("plotdot");
  
    // Convierte el SVG a una cadena de texto
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svg);
  
    // Crea un blob a partir de la cadena de texto
    const blob = new Blob([svgString], {type: "image/svg+xml;charset=utf-8"});
  
    // Crea un enlace de descarga
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "grafico.svg";
  
    // Simula un clic en el enlace para descargar el archivo
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });