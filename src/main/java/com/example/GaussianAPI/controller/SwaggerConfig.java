package com.example.GaussianAPI.controller;


import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import springfox.documentation.swagger2.annotations.EnableSwagger2;

import java.util.Collections;

@Configuration
//@EnableSwagger2
public class SwaggerConfig {
   // private static final String url = "http://194.149.135.58:8080/GaussianAPI/";
	private static final String url = "https://gaussian.chem-api.finki.ukim.mk/";
    @Bean
    public OpenAPI customConfiguration() {
        return new OpenAPI()
                .servers(Collections
                        .singletonList(new Server().url(url)))
                .components(new Components())
                .info(new Info().title("Gaussian API Docs")
                        .description("Gaussian API documentation"
                        		+"<br>"
                        		+"<br>"
                        		+ "<html><a href=" +"https://gaussian.chem-api.finki.ukim.mk/static/GaussianAPI_user_manual.html" + ">" + "Gaussian API - User manual"
                        		+"<br>"
                        		+ "<html><a href=" +"https://gaussian.chem-api.finki.ukim.mk/static/GaussianAPI-Terms_of_use.pdf" + ">" + "Gaussian API - Terms of use"
                        		+"<br>"
                        		+ "<html><a href=" + "https://gaussian.chem-api.finki.ukim.mk/static/GaussianAPI-Privacy_policy.pdf" + ">" + "Gaussian API - Privacy policy"
                        		+"<br>"
                        		+ "<html><a href=" + "https://gaussian.chem-api.finki.ukim.mk/static/GaussianAPI-Acceptable_use_policy.pdf" + ">" + "Gaussian API - Acceptable use policy"
                        		+"<br>"
                        		+"<br>"
                        		+"<br>"
                        		+"<br>"
                        		+"<html><a href=" + "https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz" + ">" + "Example input - reference data file"
                        		));
    }
}