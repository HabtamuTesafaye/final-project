let randomColor = () => {
  const borderColors = [
    'primary', 'secondary', 'success', 
    'danger', 'warning', 'info'
  ];

  return borderColors[Math.floor(Math.random() * borderColors.length)];
}

//Loading
let renderRoadProject = () => {

  var options = {
    series: [{
    name: 'Total',
    data: [360, 781.0, 20.5, 306.8, 78.7, 100.0]
  }, {
    name: 'Trunk Road Rehabilitation',
    data: [781.0, 6.5, 237.1, 523.5, 13.8, 0.1]
  },
  {
    name: 'Trunk Road Upgrading',
    data: [20.5, 0.2, 1.6, 15.3, 3.4, 0.0]
  },
  {
    name: 'New Road Project',
    data: [306.8, 1.9, 48.9, 134.2, 121.8, 0.0]
  },
  {
    name: 'Road Heavy Maintenance',
    data: [78.7, 1.3, 18.6, 54.2, 2.6, 2.0]
  },
  {
    name: 'Bridge Construction and Rehabilitation',
    data: [100.0, 0.8, 30.4, 67.0, 1.8, 0.0]
  },
],
    chart: {
    type: 'bar',
    height: 900
  },
  plotOptions: {
    bar: {
      horizontal: true,
      dataLabels: {
        position: 'top',
      },
    }
  },
  dataLabels: {
    enabled: true,
    offsetX: -6,
    style: {
      fontSize: '12px',
      colors: ['#fff']
    }
  },
  stroke: {
    show: true,
    width: 1,
    colors: ['#fff']
  },
  tooltip: {
    shared: true,
    intersect: false
  },
  xaxis: {
    categories: ["Total", "Trunk Road Rehabilitation", "Trunk Road Upgrading", "New Road Project", "Road Heavy Maintenance", "Bridge Construction and Rehabilitation"],
  },
  };

  var chart = new ApexCharts(document.querySelector("#monthly-report-graph"), options);
  chart.render();
} 

let indicatorCards = (item, seasonType, value, randomBgColor) =>{


  return `<div class="col-md-6 col-xxl-4 col-12">
            <div class="card border-${randomBgColor} ">
                <div class="card-body">
                    <div class="d-flex align-items-center">
                        <div class="flex-shrink-0">
                            <div class="avtar avtar-s  bg-light-primary">
                                <svg  width="24" height="24"
                                    viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path opacity="0.4" d="M13 9H7" stroke="#4680FF" stroke-width="1.5"
                                        stroke-linecap="round" stroke-linejoin="round" />
                                    <path
                                        d="M22.0002 10.9702V13.0302C22.0002 13.5802 21.5602 14.0302 21.0002 14.0502H19.0402C17.9602 14.0502 16.9702 13.2602 16.8802 12.1802C16.8202 11.5502 17.0602 10.9602 17.4802 10.5502C17.8502 10.1702 18.3602 9.9502 18.9202 9.9502H21.0002C21.5602 9.9702 22.0002 10.4202 22.0002 10.9702Z"
                                        stroke="#4680FF" stroke-width="1.5" stroke-linecap="round"
                                        stroke-linejoin="round" />
                                    <path
                                        d="M17.48 10.55C17.06 10.96 16.82 11.55 16.88 12.18C16.97 13.26 17.96 14.05 19.04 14.05H21V15.5C21 18.5 19 20.5 16 20.5H7C4 20.5 2 18.5 2 15.5V8.5C2 5.78 3.64 3.88 6.19 3.56C6.45 3.52 6.72 3.5 7 3.5H16C16.26 3.5 16.51 3.50999 16.75 3.54999C19.33 3.84999 21 5.76 21 8.5V9.95001H18.92C18.36 9.95001 17.85 10.17 17.48 10.55Z"
                                        stroke="#4680FF" stroke-width="1.5" stroke-linecap="round"
                                        stroke-linejoin="round" />
                                </svg>
                                </div>
                        </div>
                        <div class="flex-grow-1 ms-3" >
                            <h6 class="mb-0">${item.indicator__title_ENG} (${seasonType == 'Month' ? 'Mth' : 'Annual'}) </h6>
                        </div>
                        <div class="flex-shrink-0 ms-3">
                            <div class="dropdown"><a
                                    class="avtar avtar-s btn-link-primary dropdown-toggle arrow-none" href="#"
                                    data-bs-toggle="dropdown" aria-haspopup="true" aria-expanded="false"><i
                                        class="ti ti-dots-vertical f-18"></i></a>
                                <div class="dropdown-menu dropdown-menu-end">
                                ${seasonType == 'monthly' ? `
                                <a class="dropdown-item" href="#">Month</a>
                                <a class="dropdown-item" href="#">Year</a>
                                ` : ''}
                                   
                                    <button data-id="${item.id}"  data-type-of = "${item.indicator__type_of}" class=" detail-category  detail-category dropdown-item" data-bs-toggle="modal" data-bs-target="#exampleModal" > <svg class="pc-icon"> <use xlink:href="#custom-flash"></use></svg> Detail</button>
                                    </div>
                            </div>
                        </div>
                    </div>
                    <div class="bg-body p-3 mt-3 rounded" style="height: 100px;">
                        <div class="mt-3 row align-items-center">
                            <div class="col-7">
                                <div id="all-earnings-graph${item.indicator__id}"></div>
                            </div>
                            <div class="col-5" style="margin: -15px;">
                                    <h4 class="mb-3 ms-3 float-sm-start float-xl-end ">
                                    <span class="badge bg-light-${randomBgColor} border border-${randomBgColor} border border-success">${value[value.length - 1] ? value[value.length - 1].for_datapoint_id__year_GC : ''}  <hr>${value[value.length - 1] ? value[value.length - 1].value.toLocaleString() : ''} </span>
                                    </h4>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
          </div>`;
}

let showLoadingSpinner = (div) => {
  $(`#${div}`).html(
    `
    <div class="text-center">
  <div class="spinner-grow text-success" role="status">
    <span class="visually-hidden">Loading...</span>
  </div>
</div>
    `
  )
}

let hideLoadingSpinner = (div) =>{
  $(`#${div}`).html('')
}

let showLoadingSkeletonTopic = () => {
  for (let i = 0; i < 8; i++) {
    $("#loading-skeleton-topic").append(
      `
            <div class="col-md-6 col-xl-3 d-none d-md-block container loading-skeleton ">
               <div class="card social-widget-card">
                   <div class="card-body d-flex justify-content-between align-items-center p-2">
                    <div class="bg-body p-3 mt-3 rounded" style="height: 40px; width: 100%;">
                    </div>
                   </div>
               </div>
            </div>
            `
    );
  }
};

let hideLoadingSkeletonTopic = () => {
  $("#loading-skeleton-topic").html("");
};

let showLoadingSkeleton = () => {
  $("#loading-skeleton-category").html("")
  for (let i = 0; i < 6; i++) {
    $("#loading-skeleton-category").append(`
        <div class="col-md-4 col-xxl-4 container loading-skeleton">
            <div class="card" >
                <div class="card-body">
                    <div class="d-flex align-items-center">
                        <div class="flex-shrink-0">
                        </div>
                        <div class="flex-grow-1 ms-3" >
                            <h6 class="mb-0"></h6>
                        </div>
                    </div>
                    <div class="bg-body p-3 mt-3 rounded" style="height: 190px;">
                    <div class="text-center mt-5">
                        <div class="spinner-grow text-success" role="status">
                          <span class="visually-hidden">Loading...</span>
                          </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `);
  }
};

let hideLoadingSkeletonCategory = () => {
  $("#loading-skeleton-category").html("");
};


let renderCategoryGraph = (id, dataArray, bootstrapColorsCode) => {
  const ColorsCode = {
    "primary" : "#0d6efd",
    "secondary" : "#6610f2",
    "success" : "#198754",  // Corrected closing quotation mark
    "danger": "#dc3545",
    "warning" : "#ffc107",
    "info" : "#20c997",
};


  
  const seriesData = dataArray.map((dataPoint) => {
    return {
      x: dataPoint[0], // Year
      y: dataPoint[1], // Value
    };
  });
$(`#all-earnings-graph${id}`).html("")
new ApexCharts(document.querySelector(`#all-earnings-graph${id}`), {
    chart: {
      type: "bar",
      height: 50,
      sparkline: {
        enabled: true,
      },
    },
    colors: [
      `${ColorsCode[bootstrapColorsCode]}`,
    ],
    plotOptions: {
      bar: {
        columnWidth: "80%",
      },
    },
    series: [
      {
        data: seriesData,
      },
    ],
    xaxis: {
      crosshairs: {
        width: 1,
      },
    },
    tooltip: {
      fixed: {
        enabled: true,
      },
      x: {
        show: true,
        formatter: function (val) {
          return `Year: ${val}`; // Access x value (year) directly
        },
      },
      y: {
        title: {
          formatter: function (e) {
            return "";
          },
        },
      },
      marker: {
        show: false,
      },
    },
  }).render();
};



let renderHalfPieChart = (data, html, max) => {
  Highcharts.chart(`${html}`, {
    chart: {
        plotBackgroundColor: null,
        plotBorderWidth: 0,
        plotShadow: false
    },
    title: {
        text: `Indicator<br>shares<br>${max}`,
        align: 'center',
        verticalAlign: 'middle',
        y: 60,
        style: {
            fontSize: '1.1em'
        }
    },
    tooltip: {
        pointFormat: '{series.name}: <b>{point.percentage:.1f}%</b>'
    },
    accessibility: {
        point: {
            valueSuffix: '%'
        }
    },
    plotOptions: {
        pie: {
            dataLabels: {
                enabled: true,
                distance: -50,
                style: {
                    fontWeight: 'bold',
                    color: 'white'
                }
            },
            startAngle: -90,
            endAngle: 90,
            center: ['50%', '75%'],
            size: '110%'
        }
    },
    series: [{
        type: 'pie',
        name: 'Indicators share',
        innerSize: '50%',
        data: data
    }]
});

}


let lineGraph = (graphData, min, html) => {
  Highcharts.chart(`${html}`, {
    chart: {
        type: 'area'
    },
    title: {
        text: 'Indicators Relation'
    },
    yAxis: {
        title: {
            text: 'Value'
        }
    },
    tooltip: {
        pointFormat: '{series.name} had stockpiled <b>{point.y:,.0f}</b><br/>' +
            'warheads in {point.x}'
    },
    plotOptions: {
        area: {
            pointStart: Number(min),
            marker: {
                enabled: false,
                symbol: 'circle',
                radius: 2,
                states: {
                    hover: {
                        enabled: true
                    }
                }
            }
        }
    },
    series: graphData
  });
}

let barChart = (graphData, indicators) => {
  Highcharts.chart('barChart', {
    chart: {
        type: 'bar'
    },
    title: {
        text: `Last 3 Year's `,
        align: 'left'
    },
    xAxis: {
        categories: indicators,
        title: {
            text: null
        },
        gridLineWidth: 1,
        lineWidth: 0
    },
    yAxis: {
        min: 0,
        title: {
            text: 'Values',
            align: 'high'
        },
        labels: {
            overflow: 'justify'
        },
        gridLineWidth: 0
    },
    tooltip: {
        valueSuffix: ' '
    },
    plotOptions: {
        bar: {
            borderRadius: '50%',
            dataLabels: {
                enabled: true
            },
            groupPadding: 0.1
        }
    },
    legend: {
        layout: 'vertical',
        align: 'right',
        verticalAlign: 'top',
        x: -40,
        y: 80,
        floating: true,
        borderWidth: 1,
        backgroundColor:
            Highcharts.defaultOptions.legend.backgroundColor || '#FFFFFF',
        shadow: true
    },
    credits: {
        enabled: false
    },
    series: graphData,
  });
}

let pieChart = (graphData, max) => {
  // Data retrieved from https://netmarketshare.com/
// Build the chart
Highcharts.chart('pieChart', {
  chart: {
      plotBackgroundColor: null,
      plotBorderWidth: null,
      plotShadow: false,
      type: 'pie'
  },
  title: {
      text: `Indicator shares in ${max}`,
      align: 'left'
  },
  tooltip: {
      pointFormat: '{series.name} <b>{point.percentage:.1f}%</b>'
  },
  accessibility: {
      point: {
          valueSuffix: '%'
      }
  },
  plotOptions: {
      pie: {
          allowPointSelect: true,
          cursor: 'pointer',
          dataLabels: {
              enabled: false
          },
      }
  },
  series: [{
      name: 'Values',
      colorByPoint: true,
      data: graphData,
  }]
});

}

let fetchIndicatorDetail = () =>{
  $(".indicator-detail").click(function(){
    const buttonData = $(this).data()

      $("#analytics-tab-2")[0].click(); //Trigger to detail button clicked automatically 
  
    $.ajax({
      type: "GET",
      url: `/dashboard-api/indicator-detail/${buttonData.indicatorId}`,
      success: function(data){
        const graphData = []
        const dataValue = []
        const childData = []
        let min = data.values[0].for_datapoint_id__year_EC
        let max = data.values[data.values.length - 1].for_datapoint_id__year_EC

        let table = `
        <div class="table-responsive">
        <table class="table">
        <thead>
          <tr>
            <th scope="col">#</th>
            <th scope="col">Indicator</th>
            
        `
        for (let i = min; i <= max; i++) {
          table += `<th scope="col">${i}</th>`
        }

        table += `
      </tr>
        </thead>
        <tbody>`

        let counter = 1
        let parentIndicator = data.indicators[0]

        table += `<tr>
        <th scope="row">${counter}</th>
        <td class="fw-bold">  ${parentIndicator.title_ENG}</td>
      `
      let values = data.values.filter((value) => value.for_indicator_id == parentIndicator.id)



      for (let yr = min; yr <= max; yr++) {
        let checkedYear = false

        let yearValue = values.find((item) => item.for_datapoint_id__year_EC == yr)

          if(yearValue){
            checkedYear = true
            table += `<td>${yearValue.value}</td>`
            dataValue.push(yearValue.value)
          }
      
      if(checkedYear  == false ){
        dataValue.push(0)
        table += `<td> - </td>`
      }
    }



      graphData.push(
        {
          name: parentIndicator.title_ENG,
          data : dataValue
        }
      )

      table+=`</tr>`
       

        let filterChildIndicatorTable = (parent, space) =>{
          let spaceChild = space+"&nbsp;&nbsp;&nbsp;&nbsp;"

          let childLists = data.indicators.filter((children) => children.parent_id == parent.id )
          for(let child of  childLists){   
            if(child.parent_id == parent.id){
              counter++;
              table += `<tr>
              <th scope="row">${counter}</th>
              <td> ${space} ${child.title_ENG}</td>
            `
            let values = data.values.filter((value) => value.for_indicator_id == child.id)
            let valueYear = values.filter((value) =>value.for_datapoint_id__year_EC )
            for (let yr = min; yr <= max; yr++) {
              let checkedYear = false
      
              let yearValue = values.find((item) => item.for_datapoint_id__year_EC == yr)
      
                if(yearValue){
                  checkedYear = true
                  table += `<td>${yearValue.value}</td>`
                  dataValue.push(yearValue.value)
                }
            
            if(checkedYear  == false ){
              dataValue.push(0)
              table += `<td> - </td>`
            }
          }
            childData.push(
              [child.title_ENG, values[values.length-1].value]
            )

            table+=`</tr>`
            filterChildIndicatorTable(child, spaceChild)
            }
          }
        }
        
        filterChildIndicatorTable(parentIndicator, "&nbsp;&nbsp;")

        table += `</tbody>
        </table>
      </div>
        `

      
        $('#indicator-detail-table').html(table)
        lineGraph(graphData, min, 'lineGraph-detail')

        if(childData.length > 0){
          $('#lineGraph-detail-parent-html').removeClass('col-12').addClass('col-md-6')
          renderHalfPieChart(childData, 'half-pie-chart-detail', max)
        }else{
          $('#lineGraph-detail-parent-html').removeClass('col-md-6').addClass('col-12')

          $('#half-pie-chart-detail').html('')
        }
       
      }
    })
  })
}

let monthGraph = (data_set) =>{

  (async () => {
    /**
     * Create the chart when all data is loaded
     * @return {undefined}
     */
    function createChart(series) {
      Highcharts.stockChart("monthChart", {
        rangeSelector: {
          selected: 4,
        },

        yAxis: {
          labels: {
            format: "{#if (gt value 0)}+{/if}{value}%",
          },
          plotLines: [
            {
              value: 0,
              width: 2,
              color: "silver",
            },
          ],
        },

        plotOptions: {
          series: {
            label: {
              connectorAllowed: false,
            },
          },
        },

        tooltip: {
          pointFormat:
            '<span style="color:{series.color}">{series.name}</span>: <b>{point.y}</b><br/>',
          valueDecimals: 2,
          split: true,
        },

        series,
      });
    }
    createChart(data_set);
  })();

}

  //Handel onclick category detail
let handelCategoryDetail = () => {
    $(".detail-category").click(function () {
      let buttonData = $(this).data();
     
      $.ajax({
        url: `/dashboard-api/category_detail_list/${buttonData.id}`,
        beforeSend: function () {
          showLoadingSpinner('spinnerLoading')
          $("#analytics-tab-1")[0].click();
          $("#monthChart").hide()
          $("#analytics-tab-2").hide()
          $("#indicator-detail-table").hide()
          $("#category-detail-chart-lists").hide()
          $("#analytics-tab-2").hide()
          $("#indicator-detail-table").hide()
          $("#category-detail-chart-lists").hide()
          $('#category-detail-table').hide()
        },
        complete: function () {
          hideLoadingSpinner('spinnerLoading')
          $('#category-detail-table').show()
        },
        success: function (data) {
          data.year = data.year.sort()
          let min = data.year[0]
          let max = data.year[data.year.length - 1]

          if(buttonData.typeOf == 'yearly'){
            const graphData = []
            const barChartData = []
            const pieChartData = []
  
  
            let tableYear = `
            <div class="table-responsive">
            <table class="table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Indicator</th>
            `
            for (let i = min; i <= max; i++) {
              tableYear += `<th scope="col">${i}</th>`
            }
  
            tableYear += `
          <th scope="col">Action</th>    
          </tr>
            </thead>
            <tbody>`
            let counter = 1
            const dataValueChart1 = []
            const dataValueChart2 = []
            const dataValueChart3 = []
  
            const indicatorName = []
  
            data.indicators.forEach((item) => {
              indicatorName.push(item.title_ENG)
  
              let values = data.values.filter((value) => value.for_indicator_id == item.id)
  
  
  
  
              tableYear += `<tr>
                 <th scope="row">${counter}</th>
                  <td>${item.title_ENG}</td>
                 `
              const dataValue = []
              
  
              for (let yr = min; yr <= max; yr++) {
                let checkedYear = false

                let yearValue = values.find((item) => item.for_datapoint_id__year_EC == yr)

                  if(yearValue){
                    checkedYear = true
                    tableYear += `<td>${yearValue.value}</td>`
                    dataValue.push(yearValue.value)
                  }
              
              if(checkedYear  == false ){
                dataValue.push(0)
                tableYear += `<td> - </td>`
              }
            }
              
            tableYear+=`
              <td> <button data-indicator-id = ${item.id} class="btn btn-sm btn-primary indicator-detail" ><i class="bi bi-eye"></i></button> </td>
              </tr>`
             
  
              graphData.push(
                {
                  name: item.title_ENG,
                  data : dataValue
                }
              )
  
              dataValueChart1.push(dataValue[dataValue.length -  1])
              dataValueChart2.push(dataValue[dataValue.length -  2])
              dataValueChart3.push(dataValue[dataValue.length -  3])
  
              pieChartData.push({
                name: item.title_ENG,
                y : dataValue[dataValue.length -  1]
              })
            
  
              counter++;
            
            })
  
            barChartData.push(
              {
                name: `Year ${max}`,
                data: dataValueChart1,
              }
            )
            barChartData.push(
              {
                name: `Year ${max-1}`,
                data: dataValueChart2,
              }
            )
            barChartData.push(
              {
                name: `Year ${max-2}`,
                data: dataValueChart3,
              }
            )
  
            tableYear += `</tbody>
                </table>
              </div>
                `
            $("#analytics-tab-1")[0].click();
            $("#monthChart").hide()
            $('#category-detail-table').html(tableYear)
            $("#analytics-tab-2").show()
            $("#indicator-detail-table").show()
            $("#category-detail-chart-lists").show()
            lineGraph(graphData, min, 'lineGraph')
            barChart(barChartData,indicatorName)
            pieChart(pieChartData, max)
            fetchIndicatorDetail()
          }
          else if (buttonData.typeOf == 'monthly'){
            let data_set = []

            let table = `
            <div class="table-responsive">
            <table class="table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Year</th>
                <th scope="col">Month</th>
            `

            let counter = 1;
            for(indicator of data.indicators){
              table += `<th scope="col">${indicator.title_ENG}</th>`
            }

            table += `
          </tr>
            </thead>
            <tbody>`

            //For table creation
            for (let i = min; i <= max; i++) {
              let yearPrint = false
              

              for(let month of data.months){
                table += `
                <tr>`
                if(!yearPrint){
                  table += `
                  <th scope="col">${counter}</th>
                  <td class="fw-bold">${i}</td>
                  `
                  counter++;
                }else{
                  table += `
                  <th scope="col"></th>
                  <td> </td>
                  `
                }

                table+=`
                <td>${month.month_AMH}</td>
                `
              
                yearPrint = true


                for(indicator of data.indicators){
                   let value = data.values.find((value) => value.for_datapoint_id__year_EC == i && value.for_month_id__number == month.number && value.for_indicator_id == indicator.id)
                   if(value){
                      table+=` <td>${value.value}</td>`
                   }else{
                    table+=` <td> - </td>`
                   }
                }

                
                table+= `
                </tr>
                `
              }
              
            }


            //for chart
            let arr = [];
            for(indicator of data.indicators){
              for(let year=min; year<=max; year++){
                for(month of data.months){
                  let value = data.values.find((value) => value.for_datapoint_id__year_EC == year && value.for_month_id__number == month.number && value.for_indicator_id == indicator.id)
                  if(value){
                    arr.push([Date.UTC(parseInt(year), parseInt(month.number), 1), (value.value)]);
                  }
                }
              }
              data_set.push({'name' : indicator.title_ENG, 'data' : arr})
              arr = []
            }  

            table += `</tbody>
                  </table>
                </div>
                  `


            monthGraph(data_set) // set Chart datas
            $("#monthChart").show()
            $("#analytics-tab-1")[0].click();
            $("#analytics-tab-2").hide()
            $("#indicator-detail-table").hide()
            $("#category-detail-chart-lists").hide()
            $('#category-detail-table').html(table)
          }else if (buttonData.typeOf == 'quarterly'){s
            let data_set = []

            let table = `
            <div class="table-responsive">
            <table class="table">
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Year</th>
                <th scope="col">Quarter</th>
            `

            let counter = 1;
            for(indicator of data.indicators){
              table += `<th scope="col">${indicator.title_ENG}</th>`
            }

            table += `
          </tr>
            </thead>
            <tbody>`

            //For table creation
            for (let i = min; i <= max; i++) {
              let yearPrint = false
              

              for(let quarter of data.quarters){
                table += `
                <tr>`
                if(!yearPrint){
                  table += `
                  <th scope="col">${counter}</th>
                  <td class="fw-bold">${i}</td>
                  `
                  counter++;
                }else{
                  table += `
                  <th scope="col"></th>
                  <td> </td>
                  `
                }

                table+=`
                <td>${quarter.title_ENG}</td>
                `
              
                yearPrint = true


                for(indicator of data.indicators){
                   let value = data.values.find((value) => value.for_datapoint_id__year_EC == i && value.for_quarter__number == quarter.number && value.for_indicator_id == indicator.id)
                   if(value){
                      table+=` <td>${value.value}</td>`
                   }else{
                    table+=` <td> - </td>`
                   }
                }

                
                table+= `
                </tr>
                `
              }
              
            }


            //for chart
            let arr = [];
            for(indicator of data.indicators){
              for(let year=min; year<=max; year++){
                for(quarter of data.quarters){
                  let value = data.values.find((value) => value.for_datapoint_id__year_EC == year && value.for_quarter__number == quarter.number && value.for_indicator_id == indicator.id)
                  if(value){
                    arr.push([Date.UTC(parseInt(year), parseInt(quarter.number), 1), (value.value)]);
                  }
                }
              }
              data_set.push({'name' : indicator.title_ENG, 'data' : arr})
              arr = []
            }  

            table += `</tbody>
                  </table>
                </div>
                  `


            monthGraph(data_set) // set Chart datas
            $("#monthChart").show()
            $("#analytics-tab-1")[0].click();
            $("#analytics-tab-2").hide()
            $("#indicator-detail-table").hide()
            $("#category-detail-chart-lists").hide()
            $('#category-detail-table').html(table)
          }

        

        }

      })
    })
  }


let handleTopicClicked = () =>{
  //Handel on Click topic card
  $(".topic-card").click(function () {
    $("#bigPieChartParent").html('')
    const buttonData = $(this).data();
    
    if (buttonData.id == "Civil_Service") {
      $("#category-title").html('Civil Service');
      categoryCard = ` 
      <link
      rel="stylesheet"
      href="{% static 'dashoboard_assets/assets/css/male_female.css' %}"
      />  s
    <link
      rel="stylesheet"
      href="{% static 'dashoboard_assets/assets/css/map_style.css' %}"
    />  
    <script src="{% static 'dashoboard_assets/assets/js/pages/dashboard-default.js' %}"></script>
   
          <div class="col-lg-7 card table-responsive">
            <h3 class="text-center p-3">Regional Civil servant (2014 EFY)</h3>
            <table class="table table-bordered">
            <thead>
              <tr>
                <th>Job Grade</th>
                <th>Number of Civil Servants</th>
                <th>Share from the Total</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Professional Science (PS)</td>
                <td>39,767</td>
                <td>1.7</td>
              </tr>
              <tr>
                <td>Administrative (AD)</td>
                <td>38,736</td>
                <td>1.7</td>
              </tr>
              <tr>
                <td>Sub Professional (SP)</td>
                <td>98,080</td>
                <td>4.3</td>
              </tr>
              <tr>
                <td>Clerical and Fiscal (CF)</td>
                <td>71,895</td>
                <td>3.2</td>
              </tr>
              <tr>
                <td>Trades and Crafts (TC)</td>
                <td>27,678</td>
                <td>1.2</td>
              </tr>
              <tr>
                <td>Custodial and Manual (CM)</td>
                <td>58,892</td>
                <td>2.6</td>
              </tr>
              <tr>
                <td>Teachers</td>
                <td>78,776</td>
                <td>3.5</td>
              </tr>
              <tr>
                <td>Health Professionals</td>
                <td>119,968</td>
                <td>5.3</td>
              </tr>
              <tr>
                <td>Appointment/Cariour</td>
                <td>161,013</td>
                <td>7.1</td>
              </tr>
              <tr>
                <td>Agricultural professionals</td>
                <td>202,566</td>
                <td>8.9</td>
              </tr>
              <tr>
                <td>Engineers</td>
                <td>234,279</td>
                <td>10.3</td>
              </tr>
              <tr>
                <td>ICT Professionals</td>
                <td>163,389</td>
                <td>7.2</td>
              </tr>
              <tr>
                <td>Special Classification (SC)</td>
                <td>123,020</td>
                <td>5.4</td>
              </tr>
              <tr>
                <td>Job grade (XV-XXII)</td>
                <td>163,153</td>
                <td>7.2</td>
              </tr>
              <tr>
                <td>Not Stated (NS)</td>
                <td>536,884</td>
                <td>23.6</td>
              </tr>
              <tr>
                <td>Total</td>
                <td>2,278,770</td>
                <td>100.0</td>
              </tr>
            </tbody>
          </table>
  </div>
  <div class="col-lg-5 mt-4 mt-lg-0">
      <div class="card">
          <div class="card-body">
              <div class="d-flex align-items-center">
                  <div class="flex-shrink-0">
                      <div class="avatar avatar-s bg-light-primary">
                          <!-- avatar icon -->
                      </div>
                  </div>
                  <div class="flex-grow-1 ms-3">
                      <h6 class="mb-0">Civil Service</h6>
                  </div>
                  <div class="flex-shrink-0 ms-3">
                      <div class="dropdown">
                          <!-- dropdown menu -->
                      </div>
                  </div>
              </div>
              <div class="bg-body p-3 mt-3 rounded">
                  <div class="mt-3 row align-items-center">
                      <div class="col-12 col-lg-7">
                          <div id="all-earnings-graph3"></div>
                      </div>
                      
                  </div>
              </div>
          </div> <hr>
          <div id="bar_civil_servant"></div> <hr>
          <div id="pie_civil_servant" class="mt-5"></div>
      </div>
  </div>

  <div id="ethio_map"></div>
  
  
  <script>
  new ApexCharts(document.querySelector("#all-earnings-graph3"), {
    chart: { type: "bar", height: 80, sparkline: { enabled: !0 } },
    colors: ["#2CA87F"],
    plotOptions: { bar: { columnWidth: "80%" } },
    series: [{ data: [
      { x: "Label 1", y: 1.7 },
      { x: "Label 2", y: 1.9 },
      { x: "Label 3", y: 2.1 },
      { x: "Label 4", y: 2.2 },
      { x: "Label 5", y: 2.3 }
    ], }],
    xaxis: { crosshairs: { width: 1 } },
    tooltip: {
      fixed: { enabled: !1 },
      x: { show: !1 },
      y: {
        title: {
          formatter: function (e) {
            return "";
          },
        },
      },
      marker: { show: !1 },
    },
  }).render() 
  
  Highcharts.chart('bar_civil_servant', {
    chart: {
      type: 'bar'
    },
    title: {
      text: 'Number of Civil Servants by Region and gender'
    },
    xAxis: {
      categories: ['Tigray', 'Afar', 'Amhara', 'Oromia', 'Somali', 'Benishangul Gumuz', 'SNNP', 'Gambella', 'Harari', 'Sidama', 'South West', 'Addis Ababa', 'Dire Dawa', 'የክልል ድምር']
    },
    yAxis: {
      title: {
        text: 'Number of Civil Servants'
      }
    },
    series: [{
      name: 'Male',
      data: [63880, 28220, 241539, 397643, 102641, 22380, 188063, 19364, 6343, 66744, 49024, 72137, 6106, 1264084]
    }, {
      name: 'Female',
      data: [50757, 11966, 178514, 218781, 30120, 13405, 103864, 8455, 4884, 30497, 24615, 90290, 3888, 770036]
    }]
  });
  Highcharts.chart('pie_civil_servant', {
    chart: {
      type: 'pie'
    },
    title: {
      text: 'Civil Servants by Salary Category'
    },
    plotOptions: {
      pie: {
        innerSize: '50%',
        dataLabels: {
          enabled: true,
          format: '{point.name}: {point.y} ({point.percentage:.1f}%)'
        }
      }
    },
    series: [{
      name: 'Number of Civil Servants',
      data: [
        ['<1000', 15247],
        ['1000-1999', 229914],
        ['2000-2999', 154014],
        ['3000-3999', 253189],
        ['4000-4999', 252463],
        ['5000-5999', 251688],
        ['6000-6999', 255935],
        ['7000-7999', 263504],
        ['8000-8999', 200100],
        ['9000-9999', 155200],
        ['>10000', 189905],
        ['Not Stated', 57611]
      ]
    }]
  });
  
  Highcharts.Templating.helpers.abs = value => Math.abs(value);
  
  
  
  Highcharts.chart('male_female', {
      chart: {
          type: 'bar'
      },
      title: {
          text: 'Population pyramid for Somalia, 2021',
          align: 'left'
      },
      subtitle: {
          text: 'Source: <a ' +
              'href="https://countryeconomy.com/demography/population-structure/somalia"' +
              'target="_blank">countryeconomy.com</a>',
          align: 'left'
      },
      accessibility: {
          point: {
              valueDescriptionFormat: '{index}. Age {xDescription}, {value}%.'
          }
      },
      xAxis: [{
          categories: categories,
          reversed: false,
          labels: {
              step: 1
          },
          accessibility: {
              description: 'Age (male)'
          }
      }, { // mirror axis on right side
          opposite: true,
          reversed: false,
          categories: categories,
          linkedTo: 0,
          labels: {
              step: 1
          },
          accessibility: {
              description: 'Age (female)'
          }
      }],
      yAxis: {
          title: {
              text: null
          },
          labels: {
              format: '{abs value}%'
          },
          accessibility: {
              description: 'Percentage population',
              rangeDescription: 'Range: 0 to 5%'
          }
      },
  
      plotOptions: {
          series: {
              stacking: 'normal',
              borderRadius: '50%'
          }
      },
  
      tooltip: {
          format: '<b>{series.name}, age {point.category}</b><br/>' +
              'Population: {(abs point.y):.1f}%'
      },
  
      series: [{
          name: 'Male',
          data:  [
            63,880,
            28,220,
            241,539,
            397,643,
            102,641,
            22,380,
            188,063,
            19,364,
            6,343,
            66,744,
            49,024,
            72,137,
            6,106
        ]
      }, {
          name: 'Female',
          data: [
            50,757,
            11,966,
            178,514,
            218,781,
            30,120,
            13,405,
            103,864,
            8,455,
            4,884,
            30,497,
            24,615,
            90,290,
            3,888
        ]
      }]
  });
  </script>
  <script>
      (async () => {

        const topology = await fetch(
            'https://code.highcharts.com/mapdata/countries/et/et-all.topo.json'
        ).then(response => response.json());
    
        // Prepare demo data. The data is joined to map using value of 'hc-key'
        // property by default. See API docs for 'joinBy' for more info on linking
        // data and map.
        const data = [
            ['et-be', 35785 ], ['et-2837', 162427 ], ['et-ha', 11227 ], ['et-sn', 291927],
            ['et-ga', 27819 ], ['et-aa', 616424  ], ['et-so', 132761], ['et-dd', 9994 ],
            ['et-ti', 114637 ], ['et-af', 40186 ], ['et-am', 420053]
        ];
    
        // Create the chart
        Highcharts.mapChart('ethio_map', {
            chart: {
                map: topology
            },
    
            title: {
                text: 'Highcharts Maps basic demo'
            },
    
            subtitle: {
                text: 'Source map: <a href="http://code.highcharts.com/mapdata/countries/et/et-all.topo.json">Ethiopia</a>'
            },
    
            mapNavigation: {
                enabled: true,
                buttonOptions: {
                    verticalAlign: 'bottom'
                }
            },
    
            colorAxis: {
                min: 0
            },
    
            series: [{
                data: data,
                name: 'Civil Servant',
                states: {
                    hover: {
                        color: '#BADA55'
                    }
                },
                dataLabels: {
                    enabled: true,
                    format: '{point.name}'
                }
            }]
        });
    
    })();
    
    </script> 
  

                    
  `
      $("#category-card-list").html(categoryCard);
      hideLoadingSkeletonCategory()
    }
    else  if(buttonData.id == "project"){
      $("#category-title").html('Project');
      let categoryCard = ``;
      let dashboardItem = [
        {
        name: 'Number of Projects',
        value: 360,
        icon: '<i class="bi bi-gear-fill"></i>'
        },
        {
        name: 'Length of Road in thousand KM',
        value: 21,
        icon : '<i class="bi bi-rulers"></i>'
        },
        {
        name: 'Total Cost (original) in billion birr',
        value: 779,
        icon : '<i class="bi bi-cash-coin"></i>'
        },
        {
        name: 'Total Cost (revised) in billion birr',
        value: 781,
        icon : '<i class="bi bi-coin"></i>'
        },
        {
        name: 'Expenditure to date',
        value: 306.7,
        icon : '<i class="bi bi-calendar-fill"></i>'
        },
        {
        name: 'Total budget required to complete the project',
        value: 475,
        icon : '<i class="bi bi-cart-check-fill"></i>'
        },
        {
        name: 'Performance in percent to date (average)',
        value: 29,
        icon : '<i class="bi bi-percent"></i>'
        },
        {
        name: 'Planned average physical performance in percent for 2017 FY',
        value: 7,
        icon : '<i class="bi bi-send-fill"></i>'
        },
        {
        name: 'Budget request for 2017 FY in billion birr',
        value: 78.7,
        icon : '<i class="bi bi-wallet2"></i>'
        },
        {
        name: 'Number of Projects Not Yet Started',
        value: 137,
        icon : '<i class="bi bi-signpost"></i>'
        },
        {
        name: 'Total Estimated Cost of Projects Not Yet Started (in billion birr)',
        value: 412,
        icon : '<i class="bi bi-person-walking"></i>'
        },
        {
        name: 'Total Length of Projects Not Yet Started (in thousand KM)',
        value: 10.8,
        icon : '<i class="bi bi-easel"></i>'
        },
        {
        name: 'Budget Request for Projects Not Yet Started in the 2017 Fiscal Year (in billion birr)',
        value: 15.3,
        icon : '<i class="bi bi-cash"></i>'
        }
        ];
        let summaryItem = [
          {
           project : [
            
               ["Total", 360, 781.0, 20.5, 306.8, 78.7, 100.0],
               ["Trunk Road Rehabilitation",4, 6.5, 0.2, 1.9, 1.3, 0.8],
               ["Trunk Road Upgrading", 78, 237.1, 1.6, 48.9, 18.6, 30.4],
               ["New Road Project", 230, 523.5, 15.3, 134.2, 54.2, 67.0],
               ["Road Heavy Maintenance", 42, 13.8, 3.4, 121.8, 2.6, 1.8],
               ["Bridge Construction and Rehabilitation",6, 0.1, 0.0, 0.0, 2.0, 0.0]
           ]
          }
       ]
 

       categoryCard+=`
       <div class="col-12">
    <div class="card">
        <div class="card-header pb-0 pt-2">
            <ul class="nav nav-tabs analytics-tab" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="analytics-tab-1" data-bs-toggle="tab" data-bs-target="#analytics-tab-1-pane" type="button" role="tab" aria-controls="analytics-tab-1-pane" aria-selected="true">Road</button>
            </li> 
            <li class="nav-item" role="presentation">
            <button class="nav-link" id="analytics-tab-2" data-bs-toggle="tab" data-bs-target="#analytics-tab-2-pane" type="button" role="tab" aria-controls="analytics-tab-2-pane" aria-selected="false">Energy</button>
           </li>   
           <li class="nav-item" role="presentation">
           <button class="nav-link" id="analytics-tab-3" data-bs-toggle="tab" data-bs-target="#analytics-tab-3-pane" type="button" role="tab" aria-controls="analytics-tab-3-pane" aria-selected="false">Irrigation and Dam</button>
          </li>                
            </ul>
        </div>
        <div class="card-body">
            <div class="row">
              <div class="col-lg-9 col-xl-9 ">
                    <div class="tab-content" id="myTabContent">

                    <div class="tab-pane fade show active" id="analytics-tab-1-pane" role="tabpanel" aria-labelledby="analytics-tab-1" tabindex="0">

                        <div id="overview-chart-1" >

                        <!--Start-->

                        <div class="row">`

                        summaryItem[0].project.forEach((item) =>
                        
                        categoryCard+= `
                       
                        <div class="col-12 col-md-4">
                        <div class="card">
                            <div class="card-body">
                              <div
                                class="d-flex align-items-center justify-content-between mb-3"
                              >
                                <h5 class="mb-0">${item[0]}</h5>
                              </div>
                              <div
                                class="card rounded-4 overflow-hidden"
                                style="
                                  background-image: url('https://files.ekmcdn.com/reedycdboy/images/true-green-a4-card-290gsm-select-pack-size-40-sheets-5701-p.jpg');
                                  background-size: cover;
                                "
                              >
                                <div class="card-body">
                                  <div class="d-flex">
                                    <div class="flex-grow-1 me-3">
                                      <p class="text-white text-sm text-opacity-50 mb-0">
                                        Number of Projects
                                      </p>
                                      <h5 class="text-white">${item[1]}</h5>
                                    </div>
                                  </div>
                                  <div class="flex-grow-1 me-3">
                                    <p class="text-white text-sm text-opacity-50 mb-0">
                                      Total Cost of the project in billion birr
                                    </p>
                                    <h5 class="text-white">${item[2]}</h5>
                                  </div>
                                  <div class="row">
                                    <div class="col-auto">
                                      <p class="text-white text-sm text-opacity-50 mb-0">
                                        Total Length in Thousand KM
                                      </p>
                                      <h6 class="text-white mb-0">${item[3]}</h6>
                                    </div>
                                    <div class="col-auto">
                                      <p class="text-white text-sm text-opacity-50 mb-0">
                                        Expenditure to date in billion birr
                                      </p>
                                      <h6 class="text-white mb-0">${item[4]}</h6>
                                    </div>
                                    <div class="flex-grow-1 me-3">
                                      <p class="text-white text-sm text-opacity-50 mb-0">
                                        Total Cost of the project in billion birr
                                      </p>
                                      <h5 class="text-white">${item[5]}</h5>
                                    </div>
                                  </div>
                                </div>
                              </div>
                              <div class="text-center mt-3">
                                <h3 class="mb-1">${item[6]}</h3>
                                <p class="text-muted mb-0">Total Cost Share  in %</p>
                              </div>
                            </div>
                          </div>
                    </div>
                
                        ` )

                        

    categoryCard+=` 

                  </div>
                        <!--End-->

                          <div class="card">
                             <div class="card-body">
                              <h5 class="m-3">Road projects Regional Distribution</5>
                             <!--table-->
                             <div class="mt-3 table-responsive">
                             <table class="table table-hover">
                               <thead>
                                 <tr>
                                   <th>Region</th>
                                   <th>Number of Road Project</th>
                                   <th>Total Road Project Budget</th>
                                   <th>% share</th>
                                 </tr>
                               </thead>
                               <tbody>
                                 <tr>
                                   <td>Afar</td>
                                   <td>15</td>
                                   <td>28,298</td>
                                   <td>3.6</td>
                                 </tr>
                                 <tr>
                                   <td>Amhara</td>
                                   <td>86</td>
                                   <td>168,924</td>
                                   <td>21.6</td>
                                 </tr>
                                 <tr>
                                   <td>Amhara and Afar</td>
                                   <td>1</td>
                                   <td>2,236</td>
                                   <td>0.3</td>
                                 </tr>
                                 <tr>
                                   <td>Benishangul Gumuz</td>
                                   <td>10</td>
                                   <td>29,483</td>
                                   <td>3.8</td>
                                 </tr>
                                 <tr>
                                   <td>Dire Dawa</td>
                                   <td>1</td>
                                   <td>478</td>
                                   <td>0.1</td>
                                 </tr>
                                 <tr>
                                   <td>Dire Dawa and Somale</td>
                                   <td>2</td>
                                   <td>1,318</td>
                                   <td>0.2</td>
                                 </tr>
                                 <tr>
                                   <td>Gambella</td>
                                   <td>16</td>
                                   <td>35,258</td>
                                   <td>4.5</td>
                                 </tr>
                                 <tr>
                                   <td>Oromia</td>
                                   <td>93</td>
                                   <td>241,186</td>
                                   <td>30.9</td>
                                 </tr>
                                 <tr>
                                   <td>Sidama</td>
                                   <td>3</td>
                                   <td>4,781</td>
                                   <td>0.6</td>
                                 </tr>
                                 <tr>
                                   <td>Somale</td>
                                   <td>23</td>
                                   <td>42,467</td>
                                   <td>5.4</td>
                                 </tr>
                                 <tr>
                                   <td>South Ethiopia</td>
                                   <td>33</td>
                                   <td>82,776</td>
                                   <td>10.6</td>
                                 </tr>
                                 <tr>
                                   <td>South West</td>
                                   <td>10</td>
                                   <td>30,221</td>
                                   <td>3.9</td>
                                 </tr>
                                 <tr>
                                   <td>Tigray</td>
                                   <td>32</td>
                                   <td>47,014</td>
                                   <td>6.0</td>
                                 </tr>
                                 <tr>
                                   <td>Interregional</td>
                                   <td>35</td>
                                   <td>70,303</td>
                                   <td>9.0</td>
                                 </tr>
                               </tbody>
                               <tfoot>
                                 <tr>
                                   <th>Grand Total</th>
                                   <th>357</th>
                                   <th>781,189</th>
                                   <th>100.0</th>
                                 </tr>
                               </tfoot>
                              </table>
   
   
                             </div>
   
                           <!--end-->
                             </div>
                          </div>
                        </div>
                        

                    </div>


                    <!--Page Start -->

                    <div class="tab-pane fade" id="analytics-tab-2-pane" role="tabpanel" aria-labelledby="analytics-tab-2" tabindex="0">
                        <div id="overview-chart-2" style="min-height: 265px">

                        <!--Table Start -->
                      <div class="card">
                        <div class="card-body">
                           <h2>Energy Projects Performance </h2>
                        <div class="table-responsive"> 
                          <table class="table table-striped table-hover">
                            <thead>
                              <tr>
                                <th scope="col">#</th>
                                <th scope="col">Project Name</th>
                                <th scope="col">Location</th>
                                <th scope="col">Project Start Date</th>
                                <th scope="col">Project Expected Completion Date</th>
                                <th scope="col">Physical Performance to date</th>
                                <th scope="col">Project Expenditure to date</th>
                              </tr>
                            </thead>
                            <tbody>
                              <tr>
                                <th scope="row">1</th>
                                <td>Grand Renissance hydroelectric project</td>
                                <td>Benishangul Gumuz</td>
                                <td>2003</td>
                                <td>2017</td>
                                <td>95.79</td>
                                <td>193.3</td>
                              </tr>
                              <tr>
                                <th scope="row">2</th>
                                <td>Koisha hydroelectric project</td>
                                <td>South Western</td>
                                <td>2008</td>
                                <td>2019/2020</td>
                                <td>66.42</td>
                                <td>50.13</td>
                              </tr>
                              <tr>
                                <th scope="row">3</th>
                                <td>Aiysha Wind Energy Project</td>
                                <td>Somali</td>
                                <td>2008</td>
                                <td>2018</td>
                                <td>83.41</td>
                                <td>4.18</td>
                              </tr>
                              <tr>
                                <th scope="row">4</th>
                                <td>Aluto Geothermal energy Project</td>
                                <td>Oromia</td>
                                <td></td>
                                <td></td>
                                <td>99.39</td>
                                <td>5.38</td>
                              </tr>
                              <tr>
                                <th scope="row">5</th>
                                <td>Assela Wind Energy Project</td>
                                <td>Oromia</td>
                                <td></td>
                                <td></td>
                                <td>52.24</td>
                                <td>4.53</td>
                              </tr>
                              <tr>
                                <th scope="row">6</th>
                                <td>Bahirdar-Woldiya -Kombolcha Transmission Line</td>
                                <td>Amhara</td>
                                <td>2010</td>
                                <td>2017</td>
                                <td>91.6</td>
                                <td>6.8</td>
                              </tr>
                              <tr>
                                <th scope="row">7</th>
                                <td>Bekji and Debertabor Transmission Line</td>
                                <td>Amhara and Oromia</td>
                                <td></td>
                                <td></td>
                                <td>73.37</td>
                                <td>1.3</td>
                              </tr>
                              <tr>
                                <th scope="row">8</th>
                                <td>Southern Electric Grid Project</td>
                                <td>South Ethiopia and Sidama</td>
                                <td>2015</td>
                                <td>2017</td>
                                <td>45.45</td>
                                <td>4.15</td>
                              </tr>
                              <tr>
                                <th scope="row">9</th>
                                <td>Chaka Project</td>
                                <td>Addis Ababa</td>
                                <td></td>
                                <td></td>
                                <td>89.6</td>
                                <td>0.137</td>
                              </tr>
                              <tr>
                                <th scope="row">10</th>
                                <td>Kotebe Compound Project</td>
                                <td>Addis Ababa</td>
                                <td></td>
                                <td></td>
                                <td>96.58</td>
                                <td>0.604</td>
                              </tr>
                              <tr>
                                <th scope="row">11</th>
                                <td>Grand Renissance Collector</td>
                                <td>Benishangul Gumuz</td>
                                <td></td>
                                <td></td>
                                <td>57.2</td>
                                <td>0.109</td>
                              </tr>
                            </tbody>
                          </table>
                          </div>
                        </div>
                      </div>
                    </div>


                    </div>


                    <!-- Page End -->
                    <div
  class="tab-pane fade"
  id="analytics-tab-3-pane"
  role="tabpanel"
  aria-labelledby="analytics-tab-3"
  tabindex="0">



  <div id="overview-chart-2" style="min-height: 265px">
  <div class="card">
  <div class="card-body">
  <h2 class="p-3">Irrigation</h2>
  <div class="table-responsive">
  <table class="table table-striped table-hover">
  <thead>
    <tr>
      <th scope="col">Project</th>
      <th scope="col">Number of Projects</th>
      <th scope="col">Total Project Cost Requirements (in Billion Birr)</th>
      <th scope="col">Expenditure to date</th>
      <th scope="col">Required Cost for Project Completion (in Billion Birr)</th>
      <th scope="col">Requested Budget for 2017 FY Budget (in Billion Birr)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Total Irrigation projects</td>
      <td>32</td>
      <td>91.2</td>
      <td>28.2</td>
      <td>63.03</td>
      <td>6.6</td>
    </tr>
    <tr>
      <td>Ongoing Irrigation projects</td>
      <td>19</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>Ongoing Irrigation Projects with >50 Physical Performance</td>
      <td>10</td>
      <td>48.3</td>
      <td>24.2</td>
      <td>24.2</td>
      <td>2.30</td>
    </tr>
    <tr>
      <td>Ongoing Irrigation Projects with <50 Physical Performance</td>
      <td>9</td>
      <td>21.6</td>
      <td>4.02</td>
      <td>17.53</td>
      <td>2.4</td>
    </tr>
    <tr>
      <td>Number of Not Started Projects</td>
      <td>13</td>
      <td>21.4</td>
      <td>-</td>
      <td>21.4</td>
      <td>1.9</td>
    </tr>
  </tbody>
</table>
  </div>
  </div>
  </div>
  </div>
</div>


<div class="card">
<div class="card-body">
<h2 class="p-3">Regional Distribution of Irrigation Projects </h2>
<div class="table-responsive">
<table class="table table-striped table-hover">
  <thead>
    <tr>
      <th scope="col">Region</th>
      <th scope="col">Regional Distribution of Irrigation projects (in Number)</th>
      <th scope="col">Total Project Cost Requirements (in Billion Birr)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Harari</td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <td>Central Ethiopia</td>
      <td>1</td>
      <td>1,490</td>
    </tr>
    <tr>
      <td>Somali</td>
      <td>2</td>
      <td>8,410</td>
    </tr>
    <tr>
      <td>Benishangul Gumuz</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>Tigray</td>
      <td>2</td>
      <td>16,433</td>
    </tr>
    <tr>
      <td>Amhara</td>
      <td>6</td>
      <td>16,918</td>
    </tr>
    <tr>
      <td>Afar</td>
      <td>5</td>
      <td>7,991</td>
    </tr>
    <tr>
      <td>Oromia</td>
      <td>8</td>
      <td>30,470</td>
    </tr>
    <tr>
      <td>Oromia and Sidama</td>
      <td>1</td>
      <td>5,609</td>
    </tr>
    <tr>
      <td>South Ethiopia</td>
      <td>3</td>
      <td>2,435</td>
    </tr>
    <tr>
      <td>South Western Ethiopia</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>South Ethiopia and South Western Ethiopia</td>
      <td>1</td>
      <td>1,465</td>
    </tr>
    <tr>
      <td>Grand Total</td>
      <td>32</td>
      <td>91,221</td>
    </tr>
  </tbody>
</table>
</div>
</div>
</div>

                    </div>
                </div>



                <div class="col-lg-3">
                    <ul class="list-group list-group-flush">

                  `


        dashboardItem.forEach((item) => {
          categoryCard+= `
           <!-- Total Project -->
          <li class="list-group-item">
              <div class="d-flex align-items-start">
                  <div class="flex-shrink-0">
                      <div class="avtar avtar-s bg-light-secondary">${item.icon}
                      </div>
                  </div>
                  <div class="flex-grow-1 ms-3">
                      <div class="row g-1">
                          <div class="col-8">
                              <p class="text-muted mb-1">${item.name}</p>
                              <h4 class="mb-0">${item.value}</h4>
                          </div>
                      </div>
                  </div>
              </div>
          </li>`
        })


                        
                       
                  categoryCard+= ` </ul>
                </div>
            </div>
        </div>
    </div>
</div>
       `   
      
    
   
      $("#category-card-list").html(categoryCard);
      renderRoadProject() // render Graph
      hideLoadingSkeletonCategory()
    }
    else {

    $("#category-card-list").html("");
    $("#sidebarHtml").addClass("d-none")
    $(".selected-card").removeClass("border border-secondary shadow-lg border-4")

    const buttonData = $(this).data();
    $("#category-card-list").html("");
    $.ajax({
      type: "GET",
      url: `/dashboard-api/category_list/${buttonData.id}`,
      beforeSend: function () {
        hideLoadingSkeletonCategory();
        showLoadingSkeleton();
      },
      complete: function () {
        hideLoadingSkeletonCategory();
      }, success: function (data) {
        if(data.categories.length > 0){
          let category = ``;
          try{ $("#category-title").html(data.categories[0].dashboard_topic__title_ENG);}
          catch{ null}

          data.categories_lists.forEach((cat) => {
            $("#category-card-list").append(
              `
              <p class="fw-bold text-center" >${cat.name_ENG}</p>
              <hr class="shadow-lg p-1 rounded ">
              `
            )
  
            let filterCategoryIndicators = data.categories.filter((ind) => ind.id == cat.id)
            filterCategoryIndicators.forEach((item) => {
              const valueItem = [];
    
              let value = data.values.filter(
                (value) =>
                  value.for_indicator_id == item.indicator__id
              );
      
              
              let seasonType = value.length > 0 ? (value[0].for_indicator__type_of == "monthly" ? 'Month' : (value[0].for_indicator__type_of == "quarterly" ? "Quarter" : "Year") ) : 'None'
      
              for (val of value) {
                if (String(val.for_indicator__type_of) == "monthly") {
                  valueItem.push([
                    val.for_datapoint_id__year_GC +
                    " - " +
                    val.for_month_id__month_AMH,
                    val.value,
                  ]);
                }else if (String(val.for_indicator__type_of) == "quarterly") {
                  valueItem.push([
                    val.for_datapoint_id__year_GC +
                    " - " +
                    val.for_quarter_id__title_ENG,
                    val.value,
                  ]);
                } else {
                  valueItem.push([val.for_datapoint_id__year_GC, val.value]);
                }
              }
      
      
      
              let calculatePercentageDifference,
                roundDifference,
                difference = null;
              try {
                calculatePercentageDifference =
                  ((value[value.length - 1].value - value[value.length - 2].value) /
                    value[value.length - 2].value) *
                  100;
                roundDifference =
                  Math.round(calculatePercentageDifference * 100) / 100;
                difference = (
                  value[value.length - 1].value - value[value.length - 2].value
                ).toFixed(2);
              } catch {
                null;
              }
              let color = randomColor()
              $("#category-card-list").append(indicatorCards(item, seasonType, value,color));
              renderCategoryGraph(item.indicator__id, valueItem,color);
            });
  
          })
         
          
          handelCategoryDetail() //Call handle on category details
        }else{
          $("#category-title").html('<p class="text-center text-danger h3">No Data Found</p>')
        }
      },

    });
    }
   
  });
  
}

let defaultCategoryLists = ( search = null) => {
  //Default 
  $("#category-card-list").html("");
  $.ajax({
    type: "GET",
    url: `/dashboard-api/category_list/3${search ? '?'+search : '' }`,
    beforeSend: function () {
      showLoadingSkeleton();
    },
    complete: function () {
      hideLoadingSkeletonCategory();
    },
    success: function (data) {
      if(data.categories.length > 0){
        let category = ``;
        try{ $("#category-title").html(data.categories[0].dashboard_topic__title_ENG);}
        catch{ null}


        data.categories_lists.forEach((cat) => {
          $("#category-card-list").append(
            `
            <p class="fw-bold text-center" >${cat.name_ENG}</p>
            <hr class="shadow-lg p-1 rounded ">
            `
          )

          let filterCategoryIndicators = data.categories.filter((ind) => ind.id == cat.id)
          filterCategoryIndicators.forEach((item) => {
            const valueItem = [];
  
            let value = data.values.filter(
              (value) =>
                value.for_indicator_id == item.indicator__id
            );
    
            
            let seasonType = value.length > 0 ? (value[0].for_indicator__type_of == "monthly" ? 'Month' : (value[0].for_indicator__type_of == "quarterly" ? "Quarter" : "Year") ) : 'None'
    
            for (val of value) {
              if (String(val.for_indicator__type_of) == "monthly") {
                valueItem.push([
                  val.for_datapoint_id__year_GC +
                  " - " +
                  val.for_month_id__month_AMH,
                  val.value,
                ]);
              }else if (String(val.for_indicator__type_of) == "quarterly") {
                valueItem.push([
                  val.for_datapoint_id__year_GC +
                  " - " +
                  val.for_quarter_id__title_ENG,
                  val.value,
                ]);
              } else {
                valueItem.push([val.for_datapoint_id__year_GC, val.value]);
              }
            }
    
    
    
            let calculatePercentageDifference,
              roundDifference,
              difference = null;
            try {
              calculatePercentageDifference =
                ((value[value.length - 1].value - value[value.length - 2].value) /
                  value[value.length - 2].value) *
                100;
              roundDifference =
                Math.round(calculatePercentageDifference * 100) / 100;
              difference = (
                value[value.length - 1].value - value[value.length - 2].value
              ).toFixed(2);
            } catch {
              null;
            }
            let color = randomColor()
            $("#category-card-list").append(indicatorCards(item, seasonType, value,color));
            renderCategoryGraph(item.indicator__id, valueItem,color);
          });

        })
       
        
        handelCategoryDetail() //Call handle on category details
      }else{
        $("#category-title").html('<p class="text-center text-danger h3">No Data Found</p>')
      }
    },
  });
}


let handleOnSearch = () =>{
  $("#searchItemForm").submit(function(e){
    let form = $("#searchItemForm")
    e.preventDefault()
    let searchItem  = this.q.value
    defaultCategoryLists(`q=${searchItem}`)
  })

  $("#searchItemValue").on("keydown", function(event){
    let searchValue = $(this).val()
    // Prevent default form submission if enter is pressed (optional)
    if (event.keyCode === 13) {
      event.preventDefault();
    }

    // Minimum characters to trigger search (optional)
    if (searchValue.length < 2) {
      return; // Exit if not enough characters entered
    }

    $.ajax({
      type: "GET",
      dataType: "json",
      url: "/dashboard-api/search_category_indicator/",
      data: { search: searchValue },
      success: function(data) {
        let a = data.indicators.map((item) => `<option value="${item.indicator__title_ENG}">${item.indicator__title_ENG}</option>`)
        $("#autocomplete").html(a)
      },
    })
    
  })
}


let homePieChart = () => {
  $.ajax({
    url: "/dashboard-api/pie_chart_data",
    type: "GET",

    success: function (dataMain) {
      let chartData = [
        {
          id: '0.0',
          parent: '',
          name: "National Topic",
        },
      ];

      // Assign topics with their corresponding parent
for (const topic of dataMain.topics) {
    const newObj = {
      id: `1.${topic.id}`,
      parent: '0.0',
      name: topic.title_ENG,
      value: topic.category_count,
    };
    chartData.push(newObj);
  }
  
  // Assign categories with their corresponding parent
  for (const cat of dataMain.category) {
    const newObj = {
      id: `2.${cat.id}`,
      parent: `1.${cat.dashboard_topic__id}`,
      name: cat.name_ENG,
      value: cat.category_count,
    };
    chartData.push(newObj);
  }
  
  // Assign indicators with their corresponding parent
  for (const indicator of dataMain.indicators) {
    const obj = {
      id: `3.${indicator.id}`,
      parent: `2.${indicator.for_category__id}`,
      name: indicator.title_ENG,
      value: 3,
    };
    chartData.push(obj);
  }



  Highcharts.chart('bigPieChart', {

    chart: {
        height: '100%'
    },

    // Let the center circle be transparent
    colors: ['transparent'].concat(Highcharts.getOptions().colors),

    title: {
        text: 'National Topic'
    },

    series: [{
        type: 'sunburst',
        data: chartData,
        name: 'National Level',
        allowDrillToNode: true,
        borderRadius: 3,
        cursor: 'pointer',
        dataLabels: {
            format: '{point.name}',
            filter: {
                property: 'innerArcLength',
                operator: '>',
                value: 16
            }
        },
        levels: [{
            level: 1,
            levelIsConstant: false,
            dataLabels: {
                filter: {
                    property: 'outerArcLength',
                    operator: '>',
                    value: 64
                }
            }
        }, {
            level: 2,
            colorByPoint: true
        },
        {
            level: 3,
            colorVariation: {
                key: 'brightness',
                to: -0.5
            }
        }, {
            level: 4,
            colorVariation: {
                key: 'brightness',
                to: 0.5
            }
        }]

    }],

    tooltip: {
        headerFormat: '',
        pointFormat: '<b>{point.name}</b>' 
    }
});
    },
    error: function (xhr, status, error) {
      console.error("AJAX request failed:", error);
    },
  });
}

$(document).ready(function () {

  $.ajax({
    type: "GET",
    url: "/dashboard-api/topic_lists/",
    beforeSend: function () {
      showLoadingSkeletonTopic();
      showLoadingSkeleton();
    },
    complete: function () {
      hideLoadingSkeletonTopic();
      hideLoadingSkeletonCategory();
    },
    success: function (data) {
      handleOnSearch()
      const bootstrapColors = [
        "primary",
        "secondary",
        "success",
        "warning",
        "info",
        "dark",
      ];
      let cardTopic = ``
      let sideNav = ``;
      let selectedCard = 'border border-secondary shadow-lg border-4'
      data.topics.forEach((item) => {
        cardTopic += `
        <!-- custom cards -->
             <div class="col-md-6  col-xl-3 d-none d-md-block topic-card"
             data-id = ${item.id}
             data-category-name = "${item.title_ENG}"
             >
                <div class="card ${ item.id == 3 ? selectedCard : '' } selected-card social-widget-card bg-${bootstrapColors[Math.floor(Math.random() * bootstrapColors.length)]}">
                    <div class="card-body d-flex justify-content-between align-items-center p-2">
                        <div class="d-flex flex-column">
                            <h3 class="text-white m-0">${Number(item.category_count)} +</h3>
                            <span class="m-t-10">${item.title_ENG}</span>
                        </div>
                        <i class="fa fa-${item.icon.split(",")[1]}"></i>

                    </div>
                </div>
            </div>`;

        sideNav += `<li class="pc-item topic-card" data-id = ${item.id} data-category-name = "${item.title_ENG}">
            <a href="#" class="pc-link">
                <span class="pc-micon">
                    <i class="fa fa-${item.icon.split(",")[1]}"></i>
                </span>
                <span class="pc-mtext">${item.title_ENG}</span>   
                </a>
        </li>
        `

        
      });

       cardTopic += `
      <div class="col-md-6  col-xl-3 d-none d-md-block topic-card"  data-id="Civil_Service" data-category-name="Civil_Service" id="Civil_Service" bis_skin_checked="1">
      <div class="card  social-widget-card bg-${bootstrapColors[Math.floor(Math.random() * bootstrapColors.length)]}">
                  <div class="card-body d-flex justify-content-between align-items-center p-2" bis_skin_checked="1">
                      <div class="d-flex flex-column" bis_skin_checked="1">
                          <h3 class="text-white m-0">3 +</h3>
                          <span class="m-t-10">Civil Service</span>
                      </div>
                      <i class="fa fa-briefcase" aria-hidden="true"></i>

                  </div>
              </div>
          </div>
      `;
      cardTopic += `
      <!-- custom cards -->
      <div class="col-md-6  col-xl-3 d-none d-md-block topic-card" data-id = "project" data-category-name = "project">
         <div class="card  social-widget-card bg-${bootstrapColors[Math.floor(Math.random() * bootstrapColors.length)]}">
             <div class="card-body d-flex justify-content-between align-items-center p-2">
                 <div class="d-flex flex-column">
                     <h3 class="text-white m-0">3 +</h3>
                     <span class="m-t-10">Project</span>
                 </div>
                 <i class="fa fa-cogs" aria-hidden="true"></i>
             </div>
         </div>
     </div>
     `;

     sideNav += `  <li class="pc-item topic-card" data-id = "Civil_Service" data-category-name = "Civil_Service">
       <a href="#" class="pc-link">
           <span class="pc-micon">
           <i class="fa fa-briefcase" aria-hidden="true"></i>
           </span>
           <span class="pc-mtext">Civil Service</span>   
           </a>     
   </li>
   `
     sideNav += `  <li class="pc-item topic-card" data-id = "project" data-category-name = "project">
       <a href="#" class="pc-link">
           <span class="pc-micon">
           <i class="fa fa-cogs" aria-hidden="true"></i>
           </span>
           <span class="pc-mtext">Project</span>   
           </a>     
   </li>
   `
      $("#mobile-collapse").click(function(){
        $("#sidebarHtml").removeClass("d-none")
      })
      
      $("#topic-card-lists").html(cardTopic);
      $("#sidebar-topic-list").html(sideNav);

      handleTopicClicked() //handle Topic Clicked
      homePieChart()

    },error: function(response) {
      hideLoadingSkeletonTopic();
      hideLoadingSkeletonCategory();
    }
  
  });

  //Default 
  defaultCategoryLists()
  
});



