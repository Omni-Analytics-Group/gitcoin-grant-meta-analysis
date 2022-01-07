library(shiny)
library(shinythemes)
library(tidyverse)
library(lubridate)
library(ggthemes)
library(omnitheme)

gitcoin_grants <- read_csv("Grants Results History Round over Round + Grant over Grant - GR1-GR12.csv") %>%
    mutate(crowdfund_amount_contributions_usd = readr::parse_number(crowdfund_amount_contributions_usd)) %>%
    replace_na(list(crowdfund_amount_contributions_usd = 0))
grants_distinct <- gitcoin_grants %>%
    distinct(grant_id, grant_title) %>%
    filter(!is.na(grant_title)) %>%
    arrange(grant_title)

grant_choices <- grants_distinct$grant_id
names(grant_choices) <- grants_distinct$grant_title

ui <- navbarPage(id = "top-nav", title = div(img(src = "gc.png", width = 100)), windowTitle = "Gitcoin Grants Explorer", theme = shinytheme("cerulean"),
                 tabPanel("App", icon = icon("area-chart", "fa-2x"),

    titlePanel("Gitcoin Grant Growth Explorer"),

    sidebarLayout(
        sidebarPanel(
            selectInput("grant", "Select a Grant", choices = grant_choices)
        ),

        mainPanel(
           plotOutput("growth")
        )
    )
))

# Define server logic required to draw a histogram
server <- function(input, output) {
    
    my_data <- reactive({
        gitcoin_grants %>%
            filter(grant_id == input$grant)
    })

    output$growth <- renderPlot({
        ggplot(my_data(), aes(x = round_start_date, y = crowdfund_amount_contributions_usd)) +
            geom_line(colour = "#0F0132", size = 1.5) +
            geom_point(colour = "#11ECB6", size = 3) + 
            scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
            scale_y_continuous(labels = scales::dollar, breaks = scales::pretty_breaks(n = 10)) +
            labs(
                title = "Contributions over Time for the Selected Grant",
                subtitle = "Through Gitcoin Grants Round 12"
            ) +
            theme_fivethirtyeight() +
            theme(legend.position = "off") +
            watermark_img(filename = "gc.png", location = "center", width = 300, alpha = 0.2)
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
