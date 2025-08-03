const map = L.map('map').setView([47.899167, 17.007472], 18);

  const baseLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 20, attribution: '© OpenStreetMap'
  }).addTo(map);
  
  const orthoLayer = L.tileLayer('http://localhost:8000/{z}/{x}/{-y}.png', {
	  minZoom: 0,
	  maxZoom: 22,
	  attribution: 'ZXY Layer',
	  tms: false // Set to true if your tile folder structure follows TMS order (flipped Y)
	});

  // Add it to map
  orthoLayer.addTo(map);

  // Add Layer Switcher
  const baseMaps = {
    "OpenStreetMap": baseLayer
  };

  const overlayMaps = {
    "Orthophoto": orthoLayer,
	//"Segment Layer": segmentLayer
  };

 const layerControl = L.control.layers(baseMaps, overlayMaps, { collapsed: false }).addTo(map);
 
 const legendControl = L.control({ position: "bottomright" });

  legendControl.onAdd = function () {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML = "<b>Legend</b><br>";
    for (let key in classColors) {
      const label = document.getElementById(`label_${key}`)?.value || key;
      div.innerHTML += `<span class="legend-color" style="background:${classColors[key]}"></span>${label}<br>`;
    }
    return div;
  };

let layers = {}; // { name: { layer, geojson, type } }
let currentStyledLayer = null;
let segmentLayerName = null;
let classColors = {}, classData = {}, currentClassKey = null;
let attributeStyleMap = {};


function fixMultiPolygonNesting(geojson) {
  geojson.features = geojson.features.map(feature => {
    if (feature.geometry.type === "MultiPolygon") {
      const coords = feature.geometry.coordinates;

      // Check for nested depth, then flatten if needed
      const flattened = coords.map(polygon => {
        return polygon.map(ring => {
          // If a ring has more than 1 element and the first element isn't an array of numbers,
          // it's likely [ [ [x, y], ... ] ]
          if (Array.isArray(ring[0][0])) return ring;
          else return [ring];  // wrap in another array
        });
      });

      feature.geometry.coordinates = flattened;
    }
    return feature;
  });
  return geojson;
}

const dropZone = document.getElementById("dropZone");

dropZone.addEventListener("click", () => {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".geojson,.json,.tif,.tiff,.png,.jpg,.jpeg";
  input.onchange = e => {
    if (e.target.files.length) {
      handleFileUpload(e.target.files[0]);
    }
  };
  input.click();
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    handleFileUpload(e.dataTransfer.files[0]);
  }
});


function handleFileUpload(file) {
  const fileName = file.name.toLowerCase();
  const isImage = /\.(tif|tiff|png|jpg|jpeg)$/.test(fileName);
  const isGeoJSON = /\.(geojson|json)$/.test(fileName);

  if (isGeoJSON) {
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const raw = e.target.result;
        const geojson = JSON.parse(raw);

        if (
          !geojson.type ||
          (geojson.type !== "FeatureCollection" && geojson.type !== "Feature")
        ) {
          throw new Error("Not a valid GeoJSON object.");
        }

        const name = file.name.replace(/\.[^/.]+$/, "");
        segmentLayerIdentify(name, geojson);
      } catch (err) {
        console.error("GeoJSON parsing or loading failed:", err);
        alert("Invalid GeoJSON file.");
      }
    };
    reader.onerror = () => alert("Error reading the GeoJSON file.");
    reader.readAsText(file);
  }

  else if (isImage) {
	const name = file.name.replace(/\.[^/.]+$/, "");
    loadImageAsGeoRaster(file, name); // handle image using georaster
  }

  else {
    alert("Unsupported file type. Please upload a .geojson, .json, or image file.");
  }
}

//load geojson using relative path
function loadGeoJSONByPath() {
  const path = document.getElementById("geojsonPath").value.trim();
  const statusEl = document.getElementById("geojsonPathStatus");
  
  if (!path || !path.endsWith(".geojson")) {
    statusEl.innerText = "Please enter a valid .geojson file path.";
    statusEl.style.color = "red";
    return;
  }
  loadSegmentLayerFromFolder(path);
}


//find geojson file in the folder
function loadSegmentLayerFromFolder(url) {
    fetch(url)
      .then(res => res.json())
      .then(data => {
        const name = "Segment_Polygon";
		segmentLayerIdentify(name, data);
		console.log(data);
      })
	  .catch(err => {
		//console.warn("segment.geojson not found. Loading fallback...");
		//loadFallbackLayer();  // <-- call your alternative function here
	  });
  }
loadSegmentLayerFromFolder("results/segment.geojson");


//first layer will be segment and other layers as other overlay layers. if the results folder contains geojson with exact name, it will be used, otherwise the first uploaded file.
function segmentLayerIdentify(name, geojson){
	const type = Object.keys(layers).length === 0 ? "segment" : "viewer";
     addLayer(name, geojson, type);
	 
	 document.getElementById("url_input_geojson").style.display = "none";
	 //show button to got to samples tab
	 document.getElementById("showSamplesTabButton").style.display = "block";
}

function enableClassificationUI() {
  document.getElementById("classificationInfo").style.display = "none";
  document.getElementById("classificationTool").style.display = "block";
}


function addLayer(name, geojson, type) {
  
  //geojson = fixMultiPolygonNesting(geojson);
  //console.log(JSON.stringify(fixedGeojson.features[0].geometry, null, 2));
  //console.log(geojson);
  
  console.log("layer_type:",type, name, geojson);
  if (layers[name]) {
    alert(`Layer '${name}' already exists.`);
    return;
  }
  const layer = L.geoJSON(geojson);
  layers[name] = { layer, geojson, type };
  console.log("geojson added", layers[name])
  if (type === "segment") {
    segmentLayerName = name;
    generateClassControls();
    enableClassificationUI(true);
  }

  renderLayerList();
  showStyleOptions(name);
  
}


function renderLayerList() {
  const list = document.getElementById("layerList");
  list.innerHTML = "";
  
  // Add heading
  const heading = document.createElement("h4");
  heading.textContent = "Layers";
  list.appendChild(heading);
	
  Object.entries(layers).forEach(([name, { type }]) => {
    const div = document.createElement("div");
    div.className = "layer-item" + (name === segmentLayerName ? " active" : "");
    div.textContent = `${name} (${type})`;

    div.onclick = () => {
      // Remove 'active' from all items
      const allItems = document.querySelectorAll("#layerList .layer-item");
      allItems.forEach(item => item.classList.remove("active"));

      // Add 'active' to this item
      div.classList.add("active");

      if (layers[name].type === "segment") {
        segmentLayerName = name;
        generateClassControls();
        enableClassificationUI(true);
        renderLayerList();  // Re-render with new active state
        showStyleOptions(name);
      } else {
        showStyleOptions(name);
      }
    };

    list.appendChild(div);
  });
}


function showStyleOptions(layerName) {
  const { geojson } = layers[layerName];
  const attributes = Object.keys(geojson.features[0].properties);
  const attrSelect = document.getElementById("attributeSelect");
  const stylePanel = document.getElementById("styleConfig");
  const warning = document.getElementById("styleWarning");
  const styleMappingDiv = document.getElementById("styleMapping");

  attrSelect.innerHTML = "";
  styleMappingDiv.innerHTML = "";
  warning.style.display = "none";
  stylePanel.style.display = "block";

  attrCount =0;
  attributes.forEach(attr => {
    const opt = document.createElement("option");
    const uniqueVals = [...new Set(geojson.features.map(f => f.properties[attr]))];
    opt.value = attr;
    opt.text = `${attr} (${uniqueVals.length})`;
    opt.dataset.count = uniqueVals.length;
    attrSelect.appendChild(opt);
	
	attrCount ++;
  });

  attrSelect.onchange = () => {
    const attr = attrSelect.value;
    const uniqueVals = [...new Set(geojson.features.map(f => f.properties[attr]))];

    if (uniqueVals.length > 10) {
      warning.style.display = "block";
      warning.textContent = "Warning: The style based on attribute could not be applied. The attribute '${attr}' has ${uniqueVals.length} unique values. Please select another with fewer unique values.";
      //console.log(layers[layerName]);
	  if(!map.hasLayer(layers[layerName])){
		renderDefaultLayer(layerName, attr);  
		//console.log("inside add new layer to map")
	  }
	  
      return;
    }

    warning.style.display = "none";
    renderStyleMappingUI(layerName, attr, uniqueVals);
  };
  
  //attrSelect.selectedIndex = attrCount-1;

  attrSelect.dispatchEvent(new Event("change"));
}

function renderDefaultLayer(layerName, attr) {
   
  const { layer } = layers[layerName];
  // if (currentStyledLayer) map.removeLayer(currentStyledLayer);
  // currentStyledLayer = L.geoJSON(layer.toGeoJSON()).addTo(map);
  addStyledLayer(layerName, attr);
  
}

function generateRandomColor() {
  const letters = '0123456789ABCDEF';
  let color = '#';
  for (let i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}


function renderStyleMappingUI(layerName, attr, values) {
  const div = document.getElementById("styleMapping");
  div.innerHTML = "";
  attributeStyleMap = {};
  values.forEach(v => {
	  const color = generateRandomColor();
	  const fillColor = generateRandomColor(); // use a separate color or same if preferred

	  attributeStyleMap[v] = {
		color,
		fillColor,
		weight: 1,
		opacity: 1,
		fillOpacity: 0.3
	  };

	  const row = document.createElement("div");
	  row.className = "style-config";

	  row.innerHTML = `
		<div class="style-label"><strong>${v}</strong></div>
		<div class="style-controls">
		  <label>Color:
			<input type="color" value="${color}" data-key="${v}" class="color-input" />
		  </label>
		  <label>Fill Color:
			<input type="color" value="${fillColor}" data-key="${v}" class="fillcolor-input" />
		  </label>
		  <label>Opacity:
			<input type="number" value="1" step="0.1" min="0" max="1" data-key="${v}" class="opacity-input" />
		  </label>
		  <label>Fill Opacity:
			<input type="number" value="0.6" step="0.1" min="0" max="1" data-key="${v}" class="fillopacity-input" />
		  </label>
		  <label>Weight:
			<input type="number" value="2" step="1" min="0" max="10" data-key="${v}" class="weight-input" />
		  </label>
		</div>
	  `;

	  div.appendChild(row);
	});

	// Attach event listeners for all types
	["color", "fillcolor", "weight", "opacity", "fillopacity"].forEach(type => {
	  div.querySelectorAll(`.${type}-input`).forEach(input => {
		input.addEventListener("input", () => {
		  const key = input.dataset.key;
		  const val = input.type === "color" ? input.value : parseFloat(input.value);

		  if (type === "color") attributeStyleMap[key].color = val;
		  if (type === "fillcolor") attributeStyleMap[key].fillColor = val;
		  if (type === "weight") attributeStyleMap[key].weight = val;
		  if (type === "opacity") attributeStyleMap[key].opacity = val;
		  if (type === "fillopacity") attributeStyleMap[key].fillOpacity = val;

		  applyStyledLayer(layerName, attr);
		});
	  });
	  
	applyStyledLayer(layerName, attr);
	//console.log(layerName);
  });


  
}



function addStyledLayer(layerName, attr) {
  const { geojson, type, leafletLayer } = layers[layerName];
  if (!geojson) {
    console.warn(`No GeoJSON found for ${layerName}`);
    return;
  }

  // Remove existing styled layer if already added
  if (leafletLayer && map.hasLayer(leafletLayer)) {
    map.removeLayer(leafletLayer);
    layerControl.removeLayer(leafletLayer);
  }
	
  const styleFn = feature => {
    const val = (feature.properties[attr] || "").toString().trim();
    const style = attributeStyleMap[val] || {};

    return {
      color: style.color || "red",
      fillColor: style.fillColor || "#fff",
      weight: style.weight ?? 1,
      opacity: style.opacity ?? 1,
      fillOpacity: style.fillOpacity ?? 0
    };
  };

  const layer = L.geoJSON(geojson, {
    style: styleFn,
    pointToLayer: (f, latlng) => L.circleMarker(latlng, styleFn(f)),
    onEachFeature: type === "segment" ? onEachFeature : onEachFeatureOtherLayers
  }).addTo(map);

  layers[layerName].leafletLayer = layer;
  layerControl.addOverlay(layer, layerName);
  
   if (type === 'segment') {
    segmentLayer = layer;
    document.getElementById("classificationTool").style.display = "block";
    document.getElementById("classificationInfo").style.display = "none";
  } else {
    viewerLayer = layer;
  }
}


function applyStyledLayer(layerName, attr) {
  //console.log(layers[layerName])
  const layerObj = layers[layerName];
  if (!layerObj || !layerObj.leafletLayer) {
    console.warn(`Layer not found or not yet added: ${layerName}`);
    return;
  }

  const leafletLayer = layerObj.leafletLayer;

  leafletLayer.eachLayer(featureLayer => {
    const val = (featureLayer.feature.properties[attr] || "").toString().trim();
    const style = attributeStyleMap[val] || {};

    featureLayer.setStyle({
      color: style.color || "#000",
      fillColor: style.fillColor || "#fff",
      weight: style.weight ?? 1,
      opacity: style.opacity ?? 1,
      fillOpacity: style.fillOpacity ?? 0.3
    });
  });

  console.log(`Layer style updated: ${layerName}`);
}




function generateClassControls() {
  const count = parseInt(document.getElementById("classCount").value);
  const container = document.getElementById("classControls");
  container.innerHTML = "";
  classColors = {};
  classData = {};

  for (let i = 1; i <= count; i++) {
    const key = `class_${i}`;
    classColors[key] = generateRandomColor();
    classData[key] = [];

    const div = document.createElement("div");
	div.className = "class-row";  // Add this line
	div.innerHTML = `
	  <input type="radio" name="classRadio" value="${key}" ${i === 1 ? "checked" : ""} onchange="currentClassKey='${key}'; updateClassIds();">
	  <input type="text" id="label_${key}" value="${key}" placeholder="Class Name" oninput="updateClassIds()">
	`;
	container.appendChild(div);

  }

  currentClassKey = `class_1`;
  legendControl.addTo(map);
  updateClassIds();
}

function updateClassIds() {
  const div = document.getElementById("classwiseIds");
  div.innerHTML = "";
  for (let key in classData) {
    const ids = classData[key].sort((a, b) => a - b);
    const label = document.getElementById(`label_${key}`)?.value || key;
    const section = document.createElement("div");
    section.className = "class-section";
    section.innerHTML = `<h4>${label}</h4><div class="class-ids">${ids.join(', ') || "(none)"}</div>`;
    div.appendChild(section);
  }
  //console.log("id added");
}

function resetSelections() {
  if (!segmentLayer) return;
  segmentLayer.eachLayer(layer => {
    const fid = layer.feature.properties.segment_id;
    for (let k in classData) {
      classData[k] = classData[k].filter(id => id !== fid);
    }
    layer.setStyle({ color: "#3388ff", fillColor: "#3388ff", fillOpacity: 0 });
  });
  updateClassIds();
}


  function resetClasses() {
    document.getElementById("classControls").innerHTML = "";
    document.getElementById("classwiseIds").innerHTML = "";
    classColors = {};
    classData = {};
    currentClassKey = null;
    if (legendControl._container) legendControl.remove();
    resetSelections();
  }

function onEachFeature(feature, layer) {
//console.log("inside onEachFeature")
  const fid = feature.properties.segment_id;
  layer.on('click', () => {
	  
	//add popup
	showPropertiesInPopup(feature.properties);
	
	//console.log("clicked ddd")
    if (!currentClassKey) return alert("Please select a class first!");

    const alreadyInClass = classData[currentClassKey].includes(fid);
    for (let k in classData) {
      classData[k] = classData[k].filter(id => id !== fid);
    }

    if (!alreadyInClass) {
      classData[currentClassKey].push(fid);
      layer.setStyle({
        color: "#3388ff", //classColors[currentClassKey],
        fillColor: classColors[currentClassKey],
        fillOpacity: 0.6
      });
    } else {
      layer.setStyle({
        color: "#3388ff",
        fillColor: "#3388ff",
        fillOpacity: 0
      });
    }
	//console.log("clicked")
    updateClassIds();
	
	
	//console.log("class data", classData);
  });
}

function onEachFeatureOtherLayers(feature, layer) {
	layer.on('click', () => {
		//add popup
		showPropertiesInPopup(feature.properties);
	})
}
  
function download_class_json(){
	downloadJSON(classData, "samples.json");
}

function downloadJSON(classData, filename = "samples.json") {
  const renamedData = {};

  for (let originalKey in classData) {
    const labelInput = document.getElementById(`label_${originalKey}`);
    const newKey = labelInput?.value?.trim() || originalKey;
	console.log(newKey);

    // Merge or assign
    if (renamedData[newKey]) {
      renamedData[newKey] = renamedData[newKey].concat(classData[originalKey]);
    } else {
      renamedData[newKey] = [...classData[originalKey]];
    }
  }

  const jsonStr = JSON.stringify(renamedData, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();

  URL.revokeObjectURL(url);
}


function exportStyledGeoJSON() {
  if (!currentStyledLayer) return alert("No styled layer to export.");
  const geojson = currentStyledLayer.toGeoJSON();
  const attr = document.getElementById("attributeSelect").value;
  geojson.features.forEach(f => {
    const val = f.properties[attr];
    f.properties._style = attributeStyleMap[val];
  });
  downloadJSON(geojson, "styled_layer.geojson");
}

function exportClassifications() {
  if (!classData || Object.keys(classData).length === 0) return alert("No segment classifications available");
  const rows = [["segment_id", "class"]];
  for (let key in classData) {
    const label = document.getElementById(`label_${key}`)?.value || key;
    classData[key].forEach(id => rows.push([id, label]));
  }
  const csv = rows.map(r => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "segment_class_mapping.csv"; a.click();
  URL.revokeObjectURL(url);
}

function downloadJSON_(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}




function loadImage() {
  const input = document.getElementById("imagePath").value.trim();
  const statusEl = document.getElementById("imageLoadStatus");

  if (!input) {
    statusEl.innerText = "Please enter a tile folder URL or TMS URL.";
    statusEl.style.color = "red";
    return;
  }

  // Check if it already has ZXY pattern
  const hasZXY = input.includes("{z}") && input.includes("{x}") && (input.includes("{y}") || input.includes("{-y}"));
  let tileUrl = input;

  /// If not, append pattern
	if (!hasZXY) {
	  const endsWithSlash = input.endsWith("/");
	  const isLocalhost = input.includes("localhost") || input.includes("127.0.0.1");

	  if (isLocalhost) {
		tileUrl = endsWithSlash
		  ? `${input}{z}/{x}/{-y}.png`
		  : `${input}/{z}/{x}/{-y}.png`;
	  } else {
		tileUrl = endsWithSlash
		  ? `${input}{z}/{x}/{y}.png`
		  : `${input}/{z}/{x}/{y}.png`;
	  }
}

  // Determine if TMS
  const isTMS = tileUrl.includes("{-y}");

  try {
    // Remove existing image layer
    if (window.imageTileLayer) {
      map.removeLayer(window.imageTileLayer);
    }

    // Create and add new tile layer
    window.imageTileLayer = L.tileLayer(tileUrl, {
      minZoom: 0,
      maxZoom: 22,
      tms: isTMS,
      attribution: 'Image Tiles'
    }).addTo(map);

    statusEl.innerText = `Tile layer loaded from: ${tileUrl}`;
    statusEl.style.color = "green";
  } catch (err) {
    console.error("Error loading tile layer:", err);
    statusEl.innerText = "Failed to load tile layer.";
    statusEl.style.color = "red";
  }
}




// load image file, small size
async function loadImageAsGeoRaster(file, name) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const georaster = await parseGeoraster(arrayBuffer);
	console.log(georaster)

    // Define RGB bands (1-based index: band 3 = red, band 2 = green, band 1 = blue for true color)
    const redBand = 3;
    const greenBand = 2;
    const blueBand = 1;

    const min = georaster.mins[0];
	const max = georaster.maxs[0];

	const layer = new GeoRasterLayer({
	  georaster,
	  opacity: 1,
	  resolution: 128,
	  maxZoom: 22,
	  pixelValuesToColorFn: function(pixelValues) {
		if (pixelValues.length === 1) {
		  // single-band grayscale
		  const value = pixelValues[0];
		  const scaled = Math.round(255 * (value - min) / (max - min));
		  return `rgb(${scaled}, ${scaled}, ${scaled})`;
		} else {
		  // multi-band RGB
		  const r = pixelValues[redBand - 1];
		  const g = pixelValues[greenBand - 1];
		  const b = pixelValues[blueBand - 1];
		  return `rgb(${r}, ${g}, ${b})`;
		}
	  }
	}).addTo(map);


    layerControl.addOverlay(layer, name);
    map.fitBounds(layer.getBounds());

    document.getElementById("imageLoadStatus").innerText = "Image loaded successfully.";
  } catch (err) {
    console.error("Failed to load image as raster:", err);
    alert("Could not display raster image. Make sure it's a valid georeferenced TIFF or supported image format.");
  }
}



function showPropertiesInPopup(properties) {
  const box = document.getElementById("custom-popup-box");
  const container = box.querySelector(".popup-content");
  box.style.display = "block";

  let html = "<table>";
  for (const key in properties) {
    html += `<tr><td><strong>${key}</strong></td><td>${properties[key]}</td></tr>`;
  }
  html += "</table>";

  container.innerHTML = html;
}




const popupBox = document.getElementById('custom-popup-box');

// Stop interaction with map when scrolling or clicking inside popup
L.DomEvent.disableClickPropagation(popupBox);
L.DomEvent.disableScrollPropagation(popupBox);
popupBox.addEventListener('wheel', e => e.stopPropagation(), { passive: false });
popupBox.addEventListener('touchstart', e => e.stopPropagation(), { passive: false });


function closePopup() {
  document.getElementById("custom-popup-box").style.display = "none";
}
