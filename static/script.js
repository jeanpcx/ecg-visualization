window.addEventListener('resize', resizeSVGs);


var scatterZone = document.getElementById('scatter-zone');
var scatterRect = scatterZone.getBoundingClientRect();

// Define clusters names and colors
const clusterLabels = ["NORM", "MI", "STTC", "CD", "HYP"];
var cluster_color = ["gray", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#03305A"]
const nearby_colors = ["#03305A", 'gold', 'darkorchid', 'gray']

function clear_examine(){
    d3.selectAll(".dot").classed("examine", false);
    d3.selectAll("#nearby3 .ecg").remove();
}

// Define global dimensions and safe zones
const pageWidth = document.body.clientWidth;
const pageHeight = document.body.clientHeight;
const centerX = (pageWidth) * 0.25;
const centerY = (pageHeight) * 0.5;
const container = document.getElementById('scatter-zone');
var width = container.clientWidth;
var height = container.clientHeight;
const margin = { top: 20, right: 20, bottom: 20, left: 20 };
const margin_signal = { top: 10, right: 15, bottom: 25, left: 40 };
var graphWidth = width - margin.left - margin.right;
var graphHeight = height - margin.top - margin.bottom;

const tooltip = d3.select('.tooltip');

// Define Zoom behavior for scatter plot
let currentZoomState = d3.zoomIdentity;
let currentZoomECG = d3.zoomIdentity;
var zoom = d3.zoom()
    .scaleExtent([1, 30])
    .translateExtent([[0, 0], [width, height]])
    .on("zoom", function(event) {
        currentZoomState = event.transform;
        applyZoomTransform(event.transform);
    });

function applyZoomTransform(transform) {
    var new_x = transform.rescaleX(x);
    var new_y = transform.rescaleY(y);

    d3.select("#scatter-plot .x.axis").call(d3.axisBottom(new_x));
    d3.select("#scatter-plot .y.axis").call(d3.axisLeft(new_y));

    d3.selectAll("#scatter-plot .dot")
        .attr("cx", d => new_x(d.x))
        .attr("cy", d => new_y(d.y));
}

function zoomToPoint(d) {
    const pointX = x(d.x);
    const pointY = y(d.y);
    const scale = currentZoomState.k * 2;

    const translateX = width / 2 - pointX * scale;
    const translateY = height / 2 - pointY * scale;

    const transform = d3.zoomIdentity
        .translate(translateX, translateY)
        .scale(scale);

    svg.transition().duration(900).call(zoom.transform, transform);
}  

// Function to get color for hover buttons
function adjustSaturation(colorHex, targetSaturation) {
    let originalColor = chroma(colorHex);
    let adjustedColor = originalColor.set('hsv.s', targetSaturation);
    return adjustedColor.hex();
}

// Function to resize when window is resized
function resizeSVGs() {
    var newWidth = container.clientWidth;
    var newHeight = container.clientHeight;

    d3.select("#scatter-plot svg")
        .attr("width", newWidth)
        .attr("height", newHeight);

    x.range([0, newWidth - margin.left - margin.right]);
    y.range([newHeight - margin.top - margin.bottom, 0]);

    svg.selectAll(".dot")
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y));

    svg.call(zoom.transform, currentZoomState); // Reapply current zoom state after resize
}

// Create SVG container for scatter plot
const svg = d3.select("#scatter-plot")
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .call(zoom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

svg.append("rect")
    .attr("width", width)
    .attr("height", height)
    .style("fill", "none")
    .style("pointer-events", "all")
    .on("click", function () {
        clearECGDisplays();
    });

// Define axes x and y
const x = d3.scaleLinear().range([0, graphWidth]);
const y = d3.scaleLinear().range([graphHeight, 0]);

// Function to clear ECG displays
function clearECGDisplays() {
    d3.selectAll(".ecg").remove();
    d3.selectAll(".dot.selected").classed("selected", false);
    d3.selectAll(".dot").classed("nearby1", false);
    d3.selectAll(".dot").classed("nearby2", false);
    d3.selectAll(".dot").classed("find-nearby1", false);
    d3.selectAll(".dot").classed("find-nearby2", false);
    d3.selectAll(".dot").classed("examine", false);

    // d3.selectAll(".ecg-content").classed("graph", false);
    // d3.selectAll("#noneECG").style("display", 'block');
}


// Function to create age slider
function create_sliders(min_age, max_age){
    var ageSlider = document.getElementById('age-slider');

    // Check if the slider is already initialized
    if (ageSlider.noUiSlider) {
        console.log("Slider is already initialized. Skipping initialization.");
        return ageSlider;
    }

    // Create slider
    noUiSlider.create(ageSlider, {
        start: [40, 70],
        connect: true,
        range: { 'min': min_age, 'max': max_age},
        tooltips: true,
        format: {
            to: function (value) {return value.toFixed(0);},
            from: function (value) { return Number(value);}
        },
        pips: { mode: 'count', values: 6, density: 4}
    });

    return ageSlider // Return the d3 object
}

// Function to create buttons
var isButtonsInitialized = false
function create_buttons(clusters){
    const buttonContainer = d3.select("#button-container");

    if (isButtonsInitialized){
        console.log('Buttons are already initialized. Skipping initialization.')
        return;
    }

    // Button for All
    buttonContainer.append("button")
        .attr("class", "button cluster-button selected")
        .attr("data-cluster", clusters.length)
        .text("All")
        .style('--cluster-color', cluster_color[clusters.length]);

    // Button for each cluster
    clusters.forEach(cluster => {
        buttonContainer.append("button")
            .attr("class", "button cluster-button selected")
            .attr("data-cluster", cluster)
            .text(clusterLabels[cluster])
            .style('--cluster-color', cluster_color[cluster]);
    });
    isButtonsInitialized = true
}

// Function to get parameters amd filter data
function get_filter(data){
    var sex = d3.select(".fa.selected").attr("data-sex");
    var selectedClusters = d3.selectAll(".cluster-button.selected")
        .nodes()
        .map(button => button.getAttribute("data-cluster"));

    if (selectedClusters.length === 0) {
        return [];
    }

    if (selectedClusters.includes("5")) {
        selectedClusters = selectedClusters.slice(1);
    }

    var age_values = document.getElementById('age-slider').noUiSlider.get();
    var min = parseFloat(age_values[0]);
    var max = parseFloat(age_values[1]);

    let filtered_data = data.filter(d => {
        return (selectedClusters.includes(d.cluster.toString())) &&
               (sex === "ALL" || d.sex === sex) &&
               (d.age >= min && d.age <= max);
    });

    return filtered_data
}

// Function to manage tooltip state
function show_tooltip(d, state = true){
    if (state) {
        // var tooltipX = event.pageX;
        // var tooltipY = event.pageY;
        tooltip.classed("show", true)
            .html(d._id + "<br/><strong style='color:" + cluster_color[d.cluster] + "'>" + clusterLabels[d.cluster] + "</strong><br/>" + d.age + " years <br/>" + (d.sex === "Hombre" ? "Men" : "Women"))
            .style("left", scatterRect.left + "px")
            .style("top", scatterRect.top + "px")
            .style("--cluster-color", cluster_color[d.cluster]);
    }else{
        tooltip.classed("show", false);
    }
}

function normalize_signal(data) {
    const maxVal = d3.max(data);
    const minVal = d3.min(data);
    return data.map(d => 2 * (d - minVal) / (maxVal - minVal) - 1);
}

function smooth_signal(data, windowSize) {
    data = normalize_signal(data);
    return data.map((entry, index, array) => {
        const start = Math.max(0, index - windowSize);
        const end = Math.min(array.length, index + windowSize + 1);
        const subset = array.slice(start, end);
        const average = subset.reduce((acc, val) => acc + val, 0) / subset.length;
        return average;
    });
}

let x_signal = d3.scaleLinear().domain([0, 1000]);
let y_signal = d3.scaleLinear().domain([-1, 1]);

function draw_signal(d, signal, container_id=0){
    const container = d3.select("#nearby"+container_id);
    container.classed('graph', true);

    const signal_zone = container.append("div").attr("class", "ecg").attr("id", "ecg-" + d._id);
    const signal_plot = signal_zone.append("div").attr("class", "ecg-plot");
    const signal_info = signal_zone.append("div").attr("class", "ecg-info");

    colorNearby = (container_id == 0) ? cluster_color[d.cluster] : nearby_colors[container_id];
    signal_zone.style("--fill", colorNearby)

    // signal_zone.on("mouseover", function () {
    //     d3.select("#dot-" + d._id).classed("highlight", true);
    // }).on("mouseout", function () {
    //     d3.select("#dot-" + d._id).classed("highlight", false);
    // });

    // Show MetaData
    // signal_info.append("div").attr("class", "info-block").html(`<p>${d.id}</p>`);

    signal_info.append("div").attr("class", "info-block")
        .html(`<p style="color:white;">${d.label}</p>`)
        .style("background-color", cluster_color[d.cluster])
        .style("color", "white");
    signal_info.append("div").attr("class", "info-block").html(`<p>${(d.sex === "Hombre" ? "Men" : "Women")}</p>`);
    signal_info.append("div").attr("class", "info-block").html(`<p>${d.age} years</p>`);

    // Get local dimensions
    const plot = document.getElementsByClassName('ecg-plot')[0];
    var width_signal = plot.clientWidth;
    var height_signal = 120;

    var graphWidth_signal = width_signal - margin_signal.left - margin_signal.right;
    var graphHeight_signal = height_signal - margin_signal.top - margin_signal.bottom;

    // Create SVG for ECG
    const svg = signal_plot.append("svg")
        .attr("width", width_signal)
        .attr("height", height_signal)
        .append("g")
        .attr("transform", "translate(" + margin_signal.left + "," + margin_signal.top + ")");

    // Define clip path
    svg.append("defs").append("clipPath")
        .attr("id", "clip-" + d._id)
        .append("rect")
        .attr("width", graphWidth_signal)
        .attr("height", height_signal);

    const g = svg.append("g").attr("clip-path", "url(#clip-" + d._id + ")");

    svg.append("rect")
        .attr("width", graphWidth_signal)
        .attr("height", height_signal)
        .style("fill", "none")
        .style("pointer-events", "all")
        .call(d3.zoom()
            .scaleExtent([1, 10])
            .translateExtent([[0, 0], [graphWidth_signal, graphHeight_signal]])
            .extent([[0, 0], [graphWidth_signal, graphHeight_signal]])
            .on("zoom", (event) => {
                signal_zoom(event.transform);
            }));

    // Define axes for ECG
    x_signal.range([0, graphWidth_signal]);
    y_signal.range([graphHeight_signal, 0]);

    // Draw Signal: First lead.
    const smoothed_signal = normalize_signal(signal, 5);
    g.append("path")
        .datum(smoothed_signal)
        .attr("class", "line")
        .attr("d", d3.line().x((d, i) => x_signal(i)).y(d => y_signal(d)));

    g.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + graphHeight_signal + ")")
        .call(d3.axisBottom(x_signal));

    svg.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(y_signal));

    function signal_zoom(transform) {
        currentZoomECG = transform

        d3.selectAll(".ecg").each(function () {
            const svg = d3.select(this);

            x_signal.range([0, graphWidth_signal]);
            const xz = transform.rescaleX(x_signal);
            
            svg.select(".x.axis").call(d3.axisBottom(xz));
            svg.select(".line").attr("d", d3.line()
                .x((d, i) => xz(i))
                .y(d => y_signal(d))
            );
        });
    }
}

// Function to get signal from data, and graph
function show_signal(d, filter, renderSelected = true) {
    let filterIds = filter.map(d => d._id)
    fetch('/get_signals', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({
            id: d._id,
            filterIds: filterIds,
        }),
    })
        .then(response => response.json())
        .then(result => {
            const {nearbyIDs: nearbyIDs, selectedIds: selectedIds, signal: signals} = result;
            var ids = nearbyIDs

            console.log(result)

            if (renderSelected){
                d3.selectAll("#nearby0 .ecg").remove();
                draw_signal(d, signals[0]);
            }

            if (signals.length > 1){
                d3.selectAll(".dot").classed("nearby1", false);
                d3.selectAll(".dot").classed("nearby2", false);
                d3.selectAll(".dot").classed("find-nearby1", false);
                d3.selectAll(".dot").classed("find-nearby2", false);
                d3.selectAll("#nearby1 .ecg").remove();
                d3.selectAll("#nearby2 .ecg").remove();

                if (selectedIds != null){
                    d3.select("#dot-" + ids[1]).classed("find-nearby1", true); // Select new nearby
                    d3.select("#dot-" + ids[2]).classed("find-nearby2", true); // Select new nearby                  
                    ids = selectedIds
                }

                for (let i = 1; i < signals.length; i++) {
                    id = ids[i]
                    d3.select("#dot-" + id).classed("nearby" + i, true); // Select new nearby
                    d = filter.find(record => record._id === id);
                    
                    if (d){
                        draw_signal(d, signals[i], i);
                    }else{
                        showNotification('Oopsie! 😅 The nearest ' + i + ' point isnt within the filters', 'warning')
                    }


                }
            }
        })
        .catch((error) => {
            console.log('Error:', error);
        });
}

function examine_signal(d){
    fetch('/examine_signal', {
        // Send id of selected point to examine
        method: 'POST',
        headers: { 'Content-Type': 'application/json'},
        body: JSON.stringify({id: d._id}),
    })
        .then(response => response.json())
        .then(signal => {
            // Clean div and draw the return signal
            d3.selectAll("#nearby3 .ecg").remove();
            d3.selectAll("#nearby3").style("display", 'flex');
            d3.selectAll('.send-to-container').classed("show", true);
            draw_signal(d, signal, 3);
        })
        .catch((error) => {
            console.log('Error:', error);
        });
}

function checkSelected(filteredData){
    // Check if any point is selected
    const selected_dot = d3.select(".dot.selected");
    if (selected_dot.empty()) {
        clearECGDisplays();
    } else {
        const element = selected_dot.node();
        const data = selected_dot.data()[0];
        
        // Check if selected point is still in filtered data or if selected point if new.
        if (element.classList.contains('new-dot') || filteredData.some(d => d._id === data._id)) {
            show_signal(data, filteredData, false);
        } else {
            showNotification('Oops! 🤔 Dot not found. Adjust your filter and try again.', 'warning');
            clearECGDisplays();
        }
    }
}


function update(data, data_new) {
    let filtered_data = get_filter(data);

     // Check if any cluster is selected
    if (filtered_data.length === 0) {
        clearECGDisplays();
        svg.selectAll(".dot").remove();
        d3.select("#scatter-plot").style("display", "none");
        return;
    }else{
        d3.select("#scatter-plot").style("display", "block");
    }

    checkSelected(filtered_data);

    const dots = svg.selectAll(".fix-dot").data(filtered_data, d => d._id);
    dots.exit().remove();
    dots.attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .style("fill", d => cluster_color[d.cluster]);

    // Add new points
    dots.enter().append("circle")
        .attr("class", "dot fix-dot")
        .attr("r", 3)
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .attr("id", d => "dot-" + d._id)
        .style("fill", d => cluster_color[d.cluster])
        .on("mouseover", function (event, d) {
            show_tooltip(d);
        })
        .on("mouseout", function (event, d) {
            show_tooltip(null, false);
        })
        .on("click", function (event, d) {
            const new_dot = document.querySelectorAll('.new-dot.selected');
            if (new_dot.length == 0) {
                // If no new point is selected: continue with the normal viewing process.
                d3.selectAll(".dot.selected").classed("selected", false);
                d3.select(this).classed("selected", true);
                let filtered_data = get_filter(data);
                show_signal(d, filtered_data);
            } else {
                // If new point is selected: The user can inspect other signals.
                d3.selectAll(".dot.examine").classed("examine", false);
                d3.select(this).classed("examine", true);
                examine_signal(d);
            }
        });

    // Add new points from data_new with gray color
    const newDots = svg.selectAll(".new-dot").data(data_new, d => d._id);
    newDots.exit().remove();
    newDots.attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .style("fill", "lime");

    newDots.enter().append("circle")
        .attr("class", "new-dot dot")
        .attr("r", 3)
        .attr("cx", d => x(d.x))
        .attr("cy", d => y(d.y))
        .attr("id", d => "new-dot-" + d._id)
        .style("fill", "lime")
        .on("mouseover", function (event, d) {
            show_tooltip(d);
        })
        .on("mouseout", function (event, d) {
            show_tooltip(null, false);
        })
        .on("click", function (event, d) {
            d3.selectAll(".dot.selected").classed("selected", false);
            d3.select(this).classed("selected", true);
            let filtered_data = get_filter(data);
            show_signal(d, filtered_data);
        });
    applyZoomTransform(currentZoomState);

    d3.selectAll(".send-button").on("click", function(event) {
        var sendTo = d3.select(this).attr("send-to"); // Nearby 1 or 2?
        const examineDot = d3.select(".dot.examine")
        const d = examineDot.data()[0]; // Info of examoine point

        const selectedDot = d3.select(".new-dot.selected").data()[0];    // Info of select point
    
        if (d && selectedDot) {
            selectedDot.selected_1 = d._id;
            const name = "nearby" + sendTo;
    
            d3.selectAll(".dot." + name).classed("find-" + name, true).classed(name, false)
            examineDot.classed(name, true);
            update_nearby(selectedDot._id, sendTo, d._id);
        } else {
        console.log("No hay un punto examine o new-dot seleccionado.");
        }
    });
}


// To load data when index is upload
document.addEventListener('DOMContentLoaded', function () {
    console.log("Initializing the application...")
    fetchDataAndInitialize();
});

function fetchDataAndInitialize() {
    console.log('Loading data')
    fetch('/get_data')
        .then(response => response.json())
        .then(result => {
            const data = result.filter(row => row.cluster !== null && row.cluster !== undefined && row.cluster !== '');
            const data_new = result.filter(row => row.cluster === null || row.cluster === undefined || row.cluster === '');

            const min_age = d3.min(data, d => d.age);
            const max_age = d3.max(data, d => d.age);
            const clusters = [...new Set(data.map(d => d.cluster))];

            const ageSlider = create_sliders(min_age, max_age);
            create_buttons(clusters);

            x.domain(d3.extent(data, d => d.x));
            y.domain(d3.extent(data, d => d.y));

            update(data, data_new);

            d3.selectAll(".cluster-button").on("click", function () {
                let btn = d3.select(this);
                // All selected
                if (this.getAttribute("data-cluster") === "5") {
                    if (btn.classed("selected")) {
                        // Deselect all buttons
                        d3.selectAll('.cluster-button').classed("selected", false);
                        showNotification('Hey there! 👋 Please pick at least one cluster to continue 😊', 'alert');
                    } else {
                        // Select all buttons
                        d3.selectAll('.cluster-button').classed("selected", true);
                    }
                } else {
                    // Change status of button
                    btn.classed("selected", !btn.classed("selected"));
                    const size = d3.selectAll(".cluster-button:not([data-cluster='5']).selected").size();
                    if (size < clusters.length) {
                        // Deselect all button
                        d3.select('.cluster-button[data-cluster="5"]').classed("selected", false);
                    }
                    if (size === clusters.length) {
                        // Select all button
                        d3.select('.cluster-button[data-cluster="5"]').classed("selected", true);
                    }
                }
                update(data, data_new);
            });

            d3.selectAll("#gender-icons i").on("click", function () {
                d3.selectAll("#gender-icons i").classed("selected", false);
                d3.select(this).classed("selected", true);
                update(data, data_new);
            });

            ageSlider.noUiSlider.on('update', function (values, handle) {
                update(data, data_new);
                const selected_dot = d3.select(".dot.selected").data()[0];
                if (!selected_dot) {
                    clearECGDisplays();
                }
            });

            

        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}